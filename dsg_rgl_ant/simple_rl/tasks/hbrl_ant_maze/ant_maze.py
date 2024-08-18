import math
import random
from typing import Union
import tempfile
import networkx as nx
import numpy as np
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim, MjViewer
from scipy.spatial import distance
from skimage.draw import line_aa

from .mujoco_files.xml_generator import generate_xml
from .mujoco_model_utils import quat2mat, euler2quat, mat2euler
from enum import Enum

from ..d4rl_ant_maze.D4RLAntMazeStateClass import D4RLAntMazeState
from ...mdp.GoalDirectedMDPClass import GoalDirectedMDP


class TileType(Enum):
    EMPTY = 0
    WALL = 1
    START = 2
    REWARD = 3


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


"""
SETTINGS
"""


class AntMaze(GoalDirectedMDP):

    def __init__(self, maze_name="empty_room", image_resolution_per_tile=50, show=False, random_orientation=False,
                 fixed_goal=None, dense_reward=False, maze_scale=1):
        """
        Initialise an ant maze environment.
        THe model is automatically created using a map specification defined in /mujoco_files/maps/
        The maze_name should correspond to a map name, so /mujoco_files/maps/<maze_name>.txt should exist.
        if random_orientation=True, env.reset() with reset the agent with a random orientation. Else, agent's orientation
        if 0.
        """
        self.env_name = "ant_maze_hbrl"
        self.random_orientation = random_orientation
        self.maze_name = maze_name
        self.maze_scale = maze_scale
        self.image_resolution_per_tile = image_resolution_per_tile
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.maze_array, xml_spec_path = generate_xml(maze_name, self.temporary_directory, scale=self.maze_scale)
        self.maze_array = np.array(self.maze_array)
        self.maze_array_height, self.maze_array_width = self.maze_array.shape

        # Create Mujoco Simulation
        self.model = load_model_from_path(xml_spec_path)
        self.sim = MjSim(self.model)

        low = self.map_coordinates_to_env_position(0, -1)
        high = self.map_coordinates_to_env_position(-1, 0)
        self.maze_space = Box(low=low, high=high)  # Verify low and high
        self.goal_space = Box(low=low, high=high)
        self.goal_size = self.goal_space.shape[0]

        # Observation space
        self.state_size = len(self.sim.data.qpos) + len(self.sim.data.qvel)
        fill_size = self.state_size - self.maze_space.shape[0]
        observation_low = np.concatenate((self.maze_space.low, np.full(fill_size, float("-inf"))))
        observation_high = np.concatenate((self.maze_space.high, np.full(fill_size, float("inf"))))
        self.state_space = Box(low=observation_low, high=observation_high)

        # Action space
        self.action_space = Box(low=self.sim.model.actuator_ctrlrange[:, 0],
                                high=self.sim.model.actuator_ctrlrange[:, 1])
        self.action_size = self.action_space.shape[0]

        max_actions = 500
        num_frames_skip = 10

        self.goal_thresholds = np.array([0.5, 0.5, 0.2, 0.5, 0.5])
        self.dense_reward = dense_reward

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.show = show
        if self.show:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        self.goal = None
        self.reset()

        GoalDirectedMDP.__init__(self, range(self.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func, self.init_state,
                                 [], task_agnostic=fixed_goal is None,
                                 goal_state=fixed_goal, goal_tolerance=0.6)

    def sparse_gc_reward_function(self, state, goal, info):
        try:
            curr_pos = self.get_position(state)
            goal_pos = self.get_position(goal)
            # Re-define DSG environment reach detection
            # OLD ONE: done = np.linalg.norm(curr_pos - goal_pos) <= self.goal_tolerance
            done = (np.abs(curr_pos - goal_pos) < self.goal_thresholds[:2]).all()
        except:
            ipdb.set_trace()
        time_limit_truncated = info.get('TimeLimit.truncated', False)
        is_terminal = done and not time_limit_truncated
        reward = +0. if is_terminal else -1.
        return reward, is_terminal

    def dense_gc_reward_function(self, state, goal, info={}):
        time_limit_truncated = info.get('TimeLimit.truncated', False)
        curr_pos = self.get_position(state)
        goal_pos = self.get_position(goal)
        distance_to_goal = np.linalg.norm(curr_pos - goal_pos)
        # Re-define DSG environment reach detection
        # OLD ONE: done = distance_to_goal <= self.goal_tolerance
        done = (np.abs(curr_pos - goal_pos) < self.goal_thresholds[:2]).all()
        is_terminal = done and not time_limit_truncated
        reward = +0. if is_terminal else -distance_to_goal
        return reward, is_terminal

    def batched_sparse_gc_reward_function(self, states, goals):
        assert isinstance(states, np.ndarray)
        assert isinstance(goals, np.ndarray)

        current_positions = states[:, :2]
        goal_positions = goals[:, :2]
        distances = np.linalg.norm(current_positions - goal_positions, axis=1)
        # Re-define DSG environment reach detection
        # OLD ONE: dones = distances <= self.goal_tolerance
        dones = (np.abs(current_positions - goal_positions) < self.goal_thresholds[:2]).all(-1)

        rewards = np.zeros_like(distances)
        rewards[dones == 1] = +0.
        rewards[dones == 0] = -1.

        return rewards, dones

    def batched_dense_gc_reward_function(self, states, goals):

        current_positions = states[:, :2]
        goal_positions = goals[:, :2]

        distances = np.linalg.norm(current_positions-goal_positions, axis=1)
        # Re-define DSG environment reach detection
        # OLD ONE: dones = distances <= self.goal_tolerance
        dones = (np.abs(current_positions - goal_positions) < self.goal_thresholds[:2]).all(-1)

        assert distances.shape == dones.shape == (states.shape[0],) == (goals.shape[0],)

        rewards = -distances
        rewards[dones == 1] = 0.

        return rewards, dones

    def get_x_y_low_lims(self):
        return self.xlims[0], self.ylims[0]

    def get_x_y_high_lims(self):
        return self.xlims[1], self.ylims[1]

    def _get_state(self, observation):
        """ Convert np obs array from gym into a State object. """
        return D4RLAntMazeState(observation[:2], observation[2:], False)

    def sample_random_state(self):
        return self.sample_reachable_position()

    def sample_random_action(self):
        return self.action_space.sample()

    def _transition_func(self, state, action):
        return self.next_state

    def _reward_func(self, state, action):
        next_state, _, is_terminal = self.step(action)

        if self.task_agnostic:  # No reward function => no rewards and no terminations
            reward = 0.
            is_terminal = False
        else:
            reward, is_terminal = self.sparse_gc_reward_function(next_state, self.goal, {})
        self.next_state = self._get_state(next_state)
        return reward

    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def map_coordinates_to_env_position(self, x, y):
        x = self.maze_array_width + x if x < 0 else x
        y = self.maze_array_height + y if y < 0 else y
        res_x = (x - self.maze_array_width / 2 + 0.5) * self.maze_scale
        res_y = - (y - self.maze_array_height / 2 + 0.5) * self.maze_scale
        return np.array([res_x, res_y])

    def sample_reachable_position(self):
        # sample empty tile
        tile_coordinates = np.flip(random.choice(np.argwhere(self.maze_array != TileType.WALL.value)))

        position = self.map_coordinates_to_env_position(*tile_coordinates)

        # Sample a random position in the chosen tile
        low, high = position - 0.5, position + 0.5
        return np.random.uniform(low, high)

    # Reset simulation to state within initial state specified by user
    def reset(self):

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # Reset joint positions and velocities

        # Sample a start position
        start_tiles = np.argwhere(self.maze_array == TileType.START.value)
        start_coordinates = np.flip(random.choice(start_tiles))
        start_position = self.map_coordinates_to_env_position(*start_coordinates)

        # Build the state from it, using fixed joints and velocities values.
        orientation = random.random() * 2 * math.pi if self.random_orientation else 0
        quaternion_angle = get_quaternion_from_euler(orientation, 0, 0)
        initial_qpos = np.array([0.55] + quaternion_angle + [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        # initial_qpos = np.array([0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_qvel = np.zeros(14)
        initial_state = np.concatenate((start_position, initial_qpos, initial_qvel))

        self.sim.data.qpos[:] = initial_state[:len(self.sim.data.qpos)]
        self.sim.data.qvel[:] = initial_state[len(self.sim.data.qpos):]
        self.init_state = self._get_state(initial_state)
        self.sim.step()

        # Choose a goal. Weights are used to keep a uniform sampling over the reachable space,
        # since boxes don't cover areas of the same size.
        self.goal = self.sample_reachable_position().copy()

        # Return state
        state = self.get_state()
        return state, self.goal

    def soft_reset(self):
        """
        reset ant except orientation and position.
        """
        initial_qpos = np.array([0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_qvel = np.zeros(14)
        new_state = np.concatenate((self.sim.data.qpos[:2], initial_qpos, initial_qvel))
        new_state[:2] = self.sim.data.qpos[:2]  # Keep ant position
        new_state[4] = self.sim.data.qpos[4]  # Keep ant orientation
        self.sim.data.qpos[:] = new_state[:len(self.sim.data.qpos)]
        self.sim.data.qvel[:] = new_state[len(self.sim.data.qpos):]
        self.sim.step()

        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):
        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.show:
                self.viewer.render()
        new_state = self.get_state()

        reached = (np.abs(new_state[:len(self.goal)] - self.goal) < self.goal_thresholds[:len(self.goal)]).all()
        next_state = self.get_state()
        self.next_state = self._get_state(next_state)
        return next_state, 0 if reached else -1, reached

    def update_graph(self, graph: nx.DiGraph):
        for node_id, node_attributes in graph.nodes(data=True):
            self.set_node("node_" + str(node_id), rgba="0 1 0 1", position=node_attributes["state"])

        for edge_id, (node_1, node_2, edge_attributes) in enumerate(graph.edges(data=True)):
            first_node = nx.get_node_attributes(graph, "state")[node_1]
            second_node = nx.get_node_attributes(graph, "state")[node_2]
            self.set_edge("edge_" + str(edge_id), first_node, second_node, "0 1 0 1")

    # Visualize end goal. This function may need to be adjusted for new environments.
    def display_end_goal(self):
        self.sim.data.mocap_pos[0][:2] = np.copy(self.goal[:2])

    # Visualize all sub-goals
    def display_sub_goals(self, sub_goals):

        # Display up to 10 sub-goals and end goal
        if len(sub_goals) <= 11:
            sub_goal_ind = 0
        else:
            sub_goal_ind = len(sub_goals) - 11

        for i in range(1, min(len(sub_goals), 11)):
            self.sim.data.mocap_pos[i][:2] = np.copy(sub_goals[sub_goal_ind][:2])
            self.sim.model.site_rgba[i][3] = 1

            sub_goal_ind += 1

    def set_node(self, node_name, rgba=None, position=None):
        geom_id = self.sim.model.geom_name2id(node_name)

        if rgba is not None:
            if isinstance(rgba, str):
                rgba = np.array([float(elt) for elt in rgba.split(" ")])
            if isinstance(rgba, list):
                rgba = np.array(rgba)
            self.sim.model.geom_rgba[geom_id, :] = rgba

        if position is not None:
            if isinstance(position, list):
                position = np.array(position)
            assert len(position) <= 3
            self.sim.model.geom_pos[geom_id, :len(position)] = position

    def set_edge(self, edge_name, first_node: np.ndarray, second_node: np.ndarray, rgba=None):
        edge_id = self.sim.model.geom_name2id(edge_name)

        # Compute edge position
        position = (first_node + second_node) / 2
        self.sim.model.geom_pos[edge_id] = position

        # Compute angle between these two nodes
        diff = second_node - first_node
        angle = math.acos(diff[0] / np.linalg.norm(diff))
        euler_rotation = np.array([-math.pi / 2, angle, 0])
        self.sim.model.geom_quat[edge_id] = euler2quat(euler_rotation)



        if rgba is not None:
            if isinstance(rgba, str):
                rgba = np.array([float(elt) for elt in rgba.split(" ")])
            if isinstance(rgba, list):
                rgba = np.array(rgba)
            self.sim.model.geom_rgba[edge_id, :] = rgba

    def toggle_geom_object(self, object_name):
        geom_id = self.sim.model.geom_name2id(object_name)
        self.sim.model.geom_rgba[geom_id, -1] = 1 - self.sim.model.geom_rgba[geom_id, -1]

    def set_space_color(self, image_array: np.ndarray, low, high, color) -> np.ndarray:
        """
        Set a tile color with the given color in the given image as a numpy array of pixels
        :param image_array: The image where the tile should be set
        :param low: the box lower corner
        :param high: the box higher corner
        :param color: new color of the tile : numpy array [Red, Green, Blue]
        :return: The new image
        """
        tile_size = self.image_resolution_per_tile
        tile_width, tile_height = np.rint((high - low) * tile_size).astype(int)
        tile_img = np.tile(color, (tile_height, tile_width, 1))

        image_width, image_height, _ = image_array.shape

        # We add +1 to take images borders in consideration. Without it, x/y_min positions are computed from the
        # border of self.maze_space, that do not include maze borders.
        # NB: Y coordinates don't work the same in the state space and in the image.
        # High values (in the state space) are at the top of the image (low numpy index), with low coordinates in
        # the image. Opposite for X values.
        x_min = round((low[0] - self.maze_space.low[0] + 1) * tile_size)
        y_max = round(((image_height / tile_size) - (low[1] - self.maze_space.low[1] + 1)) * tile_size)

        x_max = x_min + tile_width
        y_min = y_max - tile_height

        image_array[y_min:y_max, x_min:x_max, :] = tile_img
        return image_array

    def render(self, ignore_goal=False) -> np.ndarray:
        """
        Return a np.ndarray of size (width, height, 3) of pixels that represent the environments and it's walls
        :return: The final image.
        """

        # we add np.full(2, 2) to get image borders, because border walls are not included in env.maze_space.
        # We assume walls have a width of one, but it's not really important for a generated image.
        image_width_px, image_height_px = np.rint((self.maze_space.high - self.maze_space.low + np.full(2, 2))
                                                  * self.image_resolution_per_tile).astype(int)

        img = np.zeros(shape=(image_height_px, image_width_px, 3), dtype=np.uint8)

        # Render the grid
        for coordinates in np.argwhere(self.maze_array != TileType.WALL.value):
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

    # Additional functions to fit with DSG code
    def state_space_size(self):
        return self.state_size

    def action_space_size(self):
        return self.action_size
