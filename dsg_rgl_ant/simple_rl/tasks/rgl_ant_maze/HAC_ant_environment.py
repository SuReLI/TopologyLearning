import os.path
from tkinter import *
from tkinter import ttk
import time
from typing import Union
from skimage.draw import line_aa
import numpy as np
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim, MjViewer
from scipy.spatial import distance


MAZE_TILE_RESOLUTION = 50  # size os a tile of the maze_grid in pixels


class HACAntEnvironment:

    def __init__(self, show=False):

        model_name = "ant_four_rooms.xml"
        self.goal_space = [[-6, 6], [-6, 6]]
        project_state_to_subgoal = lambda sim, state: state[:5]
        project_state_to_end_goal = lambda sim, state: state[:3]
        end_goal_thresholds = np.array([0.4, 0.4])

        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos, (len(initial_joint_pos), 1))
        initial_joint_ranges = np.concatenate((initial_joint_pos, initial_joint_pos), 1)
        initial_joint_ranges[0] = np.array([-6, 6])
        initial_joint_ranges[1] = np.array([-6, 6])
        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)
        subgoal_bounds = np.array([[-8, 8], [-8, 8], [0, 1], [-3, 3], [-3, 3]])
        subgoal_thresholds = np.array([0.4, 0.4, 0.2, 0.8, 0.8])

        max_actions = 1200
        num_frames_skip = 10

        self.name = model_name

        # Create Mujoco Simulation
        self.model = load_model_from_path(os.path.dirname(__file__) + "/mujoco_files/" + model_name)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        if model_name == "pendulum.xml":
            self.state_dim = 2*len(self.sim.data.qpos) + len(self.sim.data.qvel)
        else:
            self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) # State will include (i) joint angles and (ii) joint velocities
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.end_goal_dim = len(self.goal_space)
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
        self.subgoal_colors = ["Magenta","Green","Red","Blue","Cyan","Orange","Maroon","Gray","White","Black"]

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        self.goal = None
        self.maze_array = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        observation_low = np.concatenate((np.array([-8, -8, 0.45]), np.full(26, float("-inf"))))
        observation_high = np.concatenate((np.array([8, 8, 0.55]), np.full(26, float("inf"))))
        self.state_space = Box(low=observation_low, high=observation_high)

        # Action space
        self.action_space = Box(low=self.sim.model.actuator_ctrlrange[:, 0],
                                high=self.sim.model.actuator_ctrlrange[:, 1])


    # Get state, which concatenates joint positions and velocities
    def get_state(self):

        if self.name == "pendulum.xml":
            return np.concatenate([np.cos(self.sim.data.qpos),np.sin(self.sim.data.qpos),
                               self.sim.data.qvel])
        else:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    # Reset simulation to state within initial state specified by user
    def reset_sim(self, next_goal = None):

        # Reset controls
        self.sim.data.ctrl[:] = 0

        if self.name == "ant_reacher.xml":
            while True:
                # Reset joint positions and velocities
                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

                for i in range(len(self.sim.data.qvel)):
                    self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

                # Ensure initial ant position is more than min_dist away from goal
                min_dist = 8
                if np.linalg.norm(next_goal[:2] - self.sim.data.qpos[:2]) > min_dist:
                    break

        elif self.name == "ant_four_rooms.xml":

            # Choose initial start state to be different than room containing the end goal

            # Determine which of four rooms contains goal
            goal_room = 0

            if next_goal[0] < 0 and next_goal[1] > 0:
                goal_room = 1
            elif next_goal[0] < 0 and next_goal[1] < 0:
                goal_room = 2
            elif next_goal[0] > 0 and next_goal[1] < 0:
                goal_room = 3


            # Place ant in room different than room containing goal
            # initial_room = (goal_room + 2) % 4


            initial_room = np.random.randint(0,4)
            while initial_room == goal_room:
                initial_room = np.random.randint(0,4)


            # Set initial joint positions and velocities
            for i in range(len(self.sim.data.qpos)):
                self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

            # Move ant to correct room
            self.sim.data.qpos[0] = np.random.uniform(3,6.5)
            self.sim.data.qpos[1] = np.random.uniform(3,6.5)

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

            # print("Goal Room: %d" % goal_room)
            # print("Initial Ant Room: %d" % initial_room)

        else:

            # Reset joint positions and velocities
            for i in range(len(self.sim.data.qpos)):
                self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

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

        # Goal can be visualized by changing the location of the relevant site object.
        if self.name == "pendulum.xml":
            self.sim.data.mocap_pos[0] = np.array([0.5*np.sin(end_goal[0]),0,0.5*np.cos(end_goal[0])+0.6])
        elif self.name == "ur5.xml":

            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0,0.13585,0,1])
            forearm_pos_3 = np.array([0.425,0,0,1])
            wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])


            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

            # Determine joint position relative to original reference frame
            # shoulder_pos = T_1_0.dot(shoulder_pos_1)
            upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

            joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

            """
            print("\nEnd Goal Joint Pos: ")
            print("Upper Arm Pos: ", joint_pos[0])
            print("Forearm Pos: ", joint_pos[1])
            print("Wrist Pos: ", joint_pos[2])
            """

            for i in range(3):
                self.sim.data.mocap_pos[i] = joint_pos[i]

        elif self.name == "ant_reacher.xml" or self.name == "ant_four_rooms.xml":
            self.sim.data.mocap_pos[0][:2] = np.copy(end_goal[:2])

        else:
            assert False, "Provide display end goal function in environment.py file"


    # Function returns an end goal
    def get_next_goal(self,test):

        end_goal = np.zeros(len(self.goal_space))

        # Randomly select one of the four rooms in which the goal will be located
        room_num = np.random.randint(0,4)

        # Pick exact goal location
        end_goal[0] = np.random.uniform(3,6.5)
        end_goal[1] = np.random.uniform(3,6.5)

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

        return end_goal


    # Visualize all subgoals
    def display_subgoals(self,subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11


        for i in range(1,min(len(subgoals),11)):
            if self.name == "pendulum.xml":
                self.sim.data.mocap_pos[i] = np.array([0.5*np.sin(subgoals[subgoal_ind][0]),0,0.5*np.cos(subgoals[subgoal_ind][0])+0.6])
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1

            elif self.name == "ur5.xml":

                theta_1 = subgoals[subgoal_ind][0]
                theta_2 = subgoals[subgoal_ind][1]
                theta_3 = subgoals[subgoal_ind][2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0,0.13585,0,1])
                forearm_pos_3 = np.array([0.425,0,0,1])
                wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])


                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

                # Determine joint position relative to original reference frame
                # shoulder_pos = T_1_0.dot(shoulder_pos_1)
                upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

                """
                print("\nSubgoal %d Joint Pos: " % i)
                print("Upper Arm Pos: ", joint_pos[0])
                print("Forearm Pos: ", joint_pos[1])
                print("Wrist Pos: ", joint_pos[2])
                """

                # Designate site position for upper arm, forearm and wrist
                for j in range(3):
                    self.sim.data.mocap_pos[3 + 3*(i-1) + j] = np.copy(joint_pos[j])
                    self.sim.model.site_rgba[3 + 3*(i-1) + j][3] = 1

                # print("\nLayer %d Predicted Pos: " % i, wrist_1_pos[:3])

                subgoal_ind += 1

            elif self.name == "ant_reacher.xml" or self.name == "ant_four_rooms.xml":
                self.sim.data.mocap_pos[i][:3] = np.copy(subgoals[subgoal_ind][:3])
                self.sim.model.site_rgba[i][3] = 1

                subgoal_ind += 1

            else:
                # Visualize desired gripper position, which is elements 18-21 in subgoal vector
                self.sim.data.mocap_pos[i] = subgoals[subgoal_ind]
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1

    def reset(self):
        # Select final goal from final goal space, defined in "design_agent_and_env.py"
        self.goal = self.get_next_goal(test=False)  # We don't care about test, goals are sampled the same way in ant.

        # Select initial state from in initial state space, defined in environment.py
        state = self.reset_sim(self.goal)

        return state, self.goal.copy()

    def step(self, action):
        new_state = self.execute_action(action)
        assert self.goal is not None, "Environment should be reset first."

        reached = (new_state[:self.goal.shape[0]] - self.goal < self.end_goal_thresholds).all()
        reward = 0 if reached else -1
        return new_state, reward, reached

    # SOME RENDERING FUNCTIONS
    def set_space_color(self, image_array: np.ndarray, low, high, color) -> np.ndarray:
        """
        Set a tile color with the given color in the given image as a numpy array of pixels
        :param image_array: The image where the tile should be set
        :param low: the box lower corner
        :param high: the box higher corner
        :param color: new color of the tile : numpy array [Red, Green, Blue]
        :return: The new image
        """
        tile_size = MAZE_TILE_RESOLUTION
        tile_width, tile_height = np.rint((high - low) * tile_size).astype(int)
        tile_img = np.tile(color, (tile_height, tile_width, 1))

        image_width, image_height, _ = image_array.shape

        # We add +1 to take images borders in consideration. Without it, x/y_min positions are computed from the
        # border of self.maze_space, that do not include maze borders.
        # NB: Y coordinates don't work the same in the state space and in the image.
        # High values (in the state space) are at the top of the image (low numpy index), with low coordinates in
        # the image. Opposite for X values.
        x_min = round((low[0] + 8 + 1) * tile_size)
        y_max = round(((image_height / tile_size) - (low[1] + 8 + 1)) * tile_size)

        x_max = x_min + tile_width
        y_min = y_max - tile_height

        image_array[y_min:y_max, x_min:x_max, :] = tile_img
        return image_array

    def map_coordinates_to_env_position(self, x, y):
        x = self.maze_array.shape[1] + x if x < 0 else x
        y = self.maze_array.shape[0] + y if y < 0 else y
        return np.array([x - self.maze_array.shape[1] / 2 + 0.5, - (y - self.maze_array.shape[0] / 2 + 0.5)])

    def render(self, ignore_goal=False) -> np.ndarray:
        """
        Return a np.ndarray of size (width, height, 3) of pixels that represent the environments and it's walls
        :return: The final image.
        """

        # we add np.full(2, 2) to get image borders, because border walls are not included in env.maze_space.
        # We assume walls have a width of one, but it's not really important for a generated image.
        image_width_px, image_height_px = np.rint((16 + np.full(2, 2)) * MAZE_TILE_RESOLUTION).astype(int)

        img = np.zeros(shape=(image_height_px, image_width_px, 3), dtype=np.uint8)

        # Render the grid
        for coordinates in np.argwhere(self.maze_array != 1):
            coordinates = np.flip(coordinates)
            position = self.map_coordinates_to_env_position(*coordinates)
            img = self.set_space_color(img, position - 0.5, position + 0.5, np.array([255, 255, 255]))

        self.place_point(img, self.sim.data.qpos[:2], [0, 0, 255], width=10)
        if not ignore_goal:
            self.place_point(img, self.goal[:2], [255, 0, 0], width=10)
        return img

    def place_point(self, image: np.ndarray, state, color: Union[np.ndarray, list], width=5):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the state space of the point to place.
        param y: y coordinate in the state space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """
        if isinstance(color, list):
            color = np.array(color)

        state_space_range = (self.state_space.high[:2] - self.state_space.low[:2])
        center = (state[:2] - self.state_space.low[:2]) / state_space_range
        center[1] = 1 - center[1]
        center_y, center_x = (image.shape[:2] * np.flip(center)).astype(int)

        # Imagine a square of size width * width, with the coordinates computed above as a center. Iterate through
        # each pixel inside this square to
        radius = width
        for i in range(center_x - radius, center_x + radius):
            for j in range(center_y - radius, center_y + radius):
                dist = distance.euclidean((i, j), (center_x, center_y))
                if dist < radius and 0 <= i < image.shape[1] and 0 <= j < image.shape[0]:
                    image[j, i] = color
        return image

    def place_edge(self, image: np.ndarray, state_1, state_2, color: Union[np.ndarray, list], width=40):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the state space of the point to place.
        param y: y coordinate in the state space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """
        if isinstance(color, list):
            color = np.array(color)
        state_space_range = (self.state_space.high[:2] - self.state_space.low[:2])

        center = (state_1[:2] - self.state_space.low[:2]) / state_space_range
        center[1] = 1 - center[1]
        center_y_1, center_x_1 = (image.shape[:2] * np.flip(center)).astype(int)

        center = (state_2[:2] - self.state_space.low[:2]) / state_space_range[:2]
        center[1] = 1 - center[1]
        center_y_2, center_x_2 = (image.shape[:2] * np.flip(center)).astype(int)

        rr, cc, val = line_aa(center_y_1, center_x_1, center_y_2, center_x_2)
        old = image[rr, cc]
        extended_val = np.tile(val, (3, 1)).T
        image[rr, cc] = (1 - extended_val) * old + extended_val * color
