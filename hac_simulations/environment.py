import random
from copy import copy
from tkinter import *
from tkinter import ttk
import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os


class Environment:

    def __init__(self, model_name, reset_fixed, reset_coordinates, uniform_goal_sampling, maze_array, tile_size, goal_space_train,
                 goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds,
                 project_state_to_subgoal, subgoal_thresholds, max_actions=1200, num_frames_skip=10, show=False):

        self.name = model_name
        self.reset_fixed = reset_fixed
        self.uniform_goal_sampling = uniform_goal_sampling
        self.reset_coordinates = reset_coordinates
        self.maze_array = maze_array
        self.tile_size = tile_size

        # Create Mujoco Simulation
        current_file_path = os.path.dirname(__file__)
        self.model = load_model_from_path(current_file_path + "/mujoco_files/" + model_name)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) # State will include (i) joint angles and (ii) joint velocities
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:, 1] # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges

        self.end_goal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_subgoal = project_state_to_subgoal

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = ["Magenta", "Green", "Red", "Blue", "Cyan", "Orange", "Maroon", "Gray", "White", "Black"]

        self.max_actions = max_actions

        self.coordinates_normalisation_ratio = np.ones(2)

        if self.maze_array is not None:
            outside_wall_width = np.where(self.maze_array != 1)[0].min(0) * self.tile_size
            maze_width = maze_array.shape[1] * self.tile_size
            maze_height = maze_array.shape[0] * self.tile_size

            # 16 is the with of the four rooms environment HAC is tested on in the HAC paper. We assume hyper-parameters
            # has been fine-tuned for this size, so we try to make the algorithm work on this size.
            expected_size = 8 * 2
            self.coordinates_normalisation_ratio = np.array([expected_size / (maze_width - 2 * outside_wall_width),
                                                             expected_size / (maze_height - 2 * outside_wall_width)])

            maze_dims = [[- 8, 8], [- 8, 8]]

            self.goal_space_train = maze_dims
            self.goal_space_test = maze_dims
            self.subgoal_bounds_symmetric[0] = maze_dims[0][1]
            self.subgoal_bounds_symmetric[1] = maze_dims[1][1]
            self.subgoal_bounds[:2] = maze_dims
            self.end_goal_thresholds[:2] *= self.coordinates_normalisation_ratio
            self.subgoal_thresholds[:2] *= self.coordinates_normalisation_ratio

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        state = np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
        normalized_state = state[:2] * self.coordinates_normalisation_ratio
        state[:2] = normalized_state
        return state

    # Reset simulation to state within initial state specified by user
    def reset_sim(self, next_goal = None):

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # Set initial joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        if self.reset_fixed:
            # Fixed restart (the one we use for RGL)
            start_coordinates = np.flip(random.choice(np.argwhere(self.maze_array == 2)))
            x = start_coordinates[0].item()
            y = start_coordinates[1].item()
            position = np.array([x - self.maze_array.shape[0] / 2 + 0.5, - (y - self.maze_array.shape[1] / 2 + 0.5)])

            # Sample a random position in the chosen tile
            state_coords = np.random.uniform(position - 0.5, position + 0.5)
            state_coords = state_coords * self.tile_size
            self.sim.data.qpos[0:2] = state_coords[:]
        elif self.maze_array is not None:
            # Uniform restart
            start_coordinates = np.flip(random.choice(np.argwhere(self.maze_array != 1)))
            x = start_coordinates[0].item()
            y = start_coordinates[1].item()
            position = np.array([x - self.maze_array.shape[0] / 2 + 0.5, - (y - self.maze_array.shape[1] / 2 + 0.5)])

            # Sample a random position in the chosen tile
            state_coords = np.random.uniform(position - 0.5, position + 0.5)
            state_coords = state_coords * self.tile_size
            self.sim.data.qpos[0:2] = state_coords[:]

        else:
            # HAC restart
            # Choose initial start state to be different than room containing the end goal
            # Determine which of four rooms contains goal
            goal_room = 0

            if next_goal[0] < 0 and next_goal[1] > 0:
                goal_room = 1
            elif next_goal[0] < 0 and next_goal[1] < 0:
                goal_room = 2
            elif next_goal[0] > 0 and next_goal[1] < 0:
                goal_room = 3

            initial_room = np.random.randint(0,4)
            while initial_room == goal_room:
                initial_room = np.random.randint(0,4)
            # Move ant to correct room
            self.sim.data.qpos[0:2] = self.reset_coordinates.sample()[:]

            # If goal should be in top left quadrant
            if initial_room == 1:
                self.sim.data.qpos[0] *= -1

            # Else if goal should be in bottom left quadrant
            elif initial_room == 2:
                self.sim.data.qpos[0] *= -1
                self.sim.data.qpos[1] *= -1

            # Else if goal should be in bottom right quadrant
            elif initial_room == 3:
                self.sim.data.qpos[1] *= -1

        self.sim.step()

        # Return state
        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def execute_action(self, action):

        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state()


    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self,end_goal):
        self.sim.data.mocap_pos[0][:2] = np.copy(end_goal[:2])

    # Function returns an end goal
    def get_next_goal(self, test):

        if self.uniform_goal_sampling:
            tile_coordinates = np.flip(random.choice(np.argwhere(self.maze_array != 1)))
            x = tile_coordinates[0].item()
            y = tile_coordinates[1].item()
            position = np.array([x - self.maze_array.shape[0] / 2 + 0.5, - (y - self.maze_array.shape[1] / 2 + 0.5)])
            # Sample a random position in the chosen tile
            end_goal = np.random.uniform(position - 0.5, position + 0.5)
            end_goal *= self.tile_size
        else:
            end_goal = np.zeros((len(self.goal_space_test)))

            # Randomly select one of the four rooms in which the goal will be located
            room_num = np.random.randint(0,4)

            # Pick exact goal location
            end_goal[0:2] = self.reset_coordinates.sample()[:]

            # If goal should be in top left quadrant
            if room_num == 1:
                end_goal[0] *= -1

            # Else if goal should be in bottom left quadrant
            elif room_num == 2:
                end_goal[0] *= -1
                end_goal[1] *= -1

            # Else if goal should be in bottom right quadrant
            elif room_num == 3:
                end_goal[1] *= -1

        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal * self.coordinates_normalisation_ratio


    # Visualize all subgoals
    def display_subgoals(self, subgoals):
        subgoals_copy = [sub_goal.copy() for sub_goal in subgoals]
        for sub_goal in subgoals_copy:
            sub_goal[:2] /= self.coordinates_normalisation_ratio
        # Display up to 10 subgoals and end goal
        if len(subgoals_copy) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals_copy) - 11

        for i in range(1, min(len(subgoals_copy),11)):

            # Visualize desired gripper position, which is elements 18-21 in subgoal vector
            if subgoal_ind != 2:
                continue
            self.sim.data.mocap_pos[i] = subgoals_copy[subgoal_ind][:3]
            # Visualize subgoal
            self.sim.model.site_rgba[i][3] = 1
            subgoal_ind += 1
