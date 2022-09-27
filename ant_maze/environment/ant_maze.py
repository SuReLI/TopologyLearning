import random
from random import choice, choices

import numpy as np
from gym.spaces import Box
from mujoco_py import load_model_from_path, MjSim, MjViewer
from scipy.spatial import distance
from ant_maze.environment.mujoco_files.xml_generator import generate_xml

"""
SETTINGS
"""

MAZE_TILE_RESOLUTION = 50  # size os a tile of the maze_grid in pixels


class AntMaze:

    def __init__(self, maze_name="empty_room", image_resolution_per_tile=50, show=False):
        """
        Initialise an ant maze environment.
        THe model is automatically created using a map specification defined in /mujoco_files/maps/
        The maze_name should correspond to a map name, so /mujoco_files/maps/<maze_name>.txt should exist.
        """
        self.maze_name = maze_name
        self.image_resolution_per_tile = image_resolution_per_tile
        maze_info, xml_spec_path = generate_xml(maze_name)  # TODO : check maze_info["reachable_spaces_size"]

        # Create Mujoco Simulation
        self.model = load_model_from_path(xml_spec_path)
        self.sim = MjSim(self.model)

        self.initial_spaces = maze_info["initial_spaces"]
        self.reachable_spaces = maze_info["reachable_spaces"]
        self.reachable_spaces_areas = maze_info["reachable_spaces_size"]

        low = self.reachable_spaces[0].low
        high = self.reachable_spaces[0].high
        for reachable_space in self.reachable_spaces:
            low = np.minimum(low, reachable_space.low)
            high = np.maximum(high, reachable_space.high)
        self.maze_space = Box(low=low, high=high)
        self.goal_space = Box(low=np.append(low, 0.45), high=np.append(high, 0.55))
        self.goal_size = self.goal_space.shape[0]

        # Observation space
        self.observation_size = len(self.sim.data.qpos) + len(self.sim.data.qvel)
        fill_size = self.observation_size - self.maze_space.shape[0]
        observation_low = np.concatenate((self.maze_space.low, np.full(fill_size, float("-inf"))))
        observation_high = np.concatenate((self.maze_space.high, np.full(fill_size, float("inf"))))
        self.observation_space = Box(low=observation_low, high=observation_high)

        # Action space
        self.action_space = Box(low=self.sim.model.actuator_ctrlrange[:, 0],
                                high=self.sim.model.actuator_ctrlrange[:, 1])
        self.action_size = self.action_space.shape[0]

        max_actions = 500
        num_frames_skip = 10

        self.goal_thresholds = np.array([0.5, 0.5, 0.2, 0.5, 0.5])

        self.max_actions = max_actions

        # Implement visualization if necessary
        self.show = show
        if self.show:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        self.goal = None

    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # Reset joint positions and velocities
        # Sample a start area,
        start_area = choice(self.initial_spaces)

        # Sample a start position
        start_position = start_area.sample()

        # Build the state from it, using fixed joints and velocities values.
        angle = random.random() * 2 - 1  # Sample random orientation for the beginning
        initial_qpos = np.array([0.55, 1.0, 0.0, 0.0, angle, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_qvel = np.zeros(14)
        state = np.concatenate((start_position, initial_qpos, initial_qvel))

        self.sim.data.qpos[:] = state[:len(self.sim.data.qpos)]
        self.sim.data.qvel[:] = state[len(self.sim.data.qpos):]
        self.sim.step()

        # Choose a goal. Weights are used to keep a uniform sampling over the reachable space,
        # since boxes don't cover areas of the same size.
        goal_position = choices(self.reachable_spaces, weights=self.reachable_spaces_areas)[0].sample()
        torso_height = np.random.uniform(0.45, 0.55, (1,))
        self.goal = np.concatenate((goal_position, torso_height))
        if self.show:
            self.sim.data.mocap_pos[0][:2] = np.copy(self.goal[:2])

        if self.show:
            self.viewer.render()
        # Return state
        return self.get_state(), self.goal

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):
        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.show:
                self.viewer.render()
        new_state = self.get_state()
        reached = ((new_state[:len(self.goal)] - self.goal) < self.goal_thresholds[:len(self.goal)]).all()
        return self.get_state(), 0 if reached else -1, reached

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

    def set_space_color(self, image_array: np.ndarray, box, color) -> np.ndarray:
        """
        Set a tile color with the given color in the given image as a numpy array of pixels
        :param image_array: The image where the tile should be set
        :param box: the box that represent the space to color
        :param color: new color of the tile : numpy array [Red, Green, Blue]
        :return: The new image
        """
        tile_size = MAZE_TILE_RESOLUTION
        tile_width, tile_height = np.rint((box.high - box.low) * tile_size).astype(int)
        tile_img = np.tile(color, (tile_height, tile_width, 1))

        image_width, image_height, _ = image_array.shape

        # We add +1 to take images borders in consideration. Without it, x/y_min positions are computed from the
        # border of self.maze_space, that do not include maze borders.
        # NB: Y coordinates don't work the same in the state space and in the image.
        # High values (in the state space) are at the top of the image (low numpy index), with low coordinates in
        # the image. Opposite for X values.
        x_min = round((box.low[0] - self.maze_space.low[0] + 1) * tile_size)
        y_max = round(((image_height / tile_size) - (box.low[1] - self.maze_space.low[1] + 1)) * tile_size)

        x_max = x_min + tile_width
        y_min = y_max - tile_height

        image_array[y_min:y_max, x_min:x_max, :] = tile_img
        return image_array

    def render(self) -> np.ndarray:
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
        for space in self.reachable_spaces:
            img = self.set_space_color(img, space, np.array([255, 255, 255]))
        return img

    def place_point(self, image: np.ndarray, state, color: np.ndarray, width=50):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the state space of the point to place.
        param y: y coordinate in the state space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """
        x, y = tuple(state[:2])
        tile_size = MAZE_TILE_RESOLUTION

        # Convert x, y state_space_position into maze_coordinates to get point center
        # Explanation: same as in self.set_space_color, lines of code 5/6
        x_center_px = round((x - self.maze_space.low[0] + 1) * tile_size)
        y_center_px = round(((image.shape[1] / tile_size) - (y - self.maze_space.low[1] + 1)) * tile_size)

        # Imagine a square of size width * width, with the coordinates computed above as a center. Iterate through
        # each pixel inside this square to
        radius = round(width / 2) + 1
        for i in range(x_center_px - radius, x_center_px + radius):
            for j in range(y_center_px - radius, y_center_px + radius):
                if distance.euclidean((i + 0.5, j + 0.5), (x_center_px, y_center_px)) < radius:
                    image[j, i] = color

    def place_edge(self, image: np.ndarray, state_1, state_2, color: np.ndarray, width=40):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the state space of the point to place.
        param y: y coordinate in the state space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """
        tile_size = MAZE_TILE_RESOLUTION
        x1, y1 = tuple(state_1[:2])
        x2, y2 = tuple(state_2[:2])

        # Convert x, y state_space_position into maze_coordinates to get point center
        # Explanation: same as in self.set_space_color, lines of code 5/6
        x1_center_px = (x1 - self.maze_space.low[0] + 1) * tile_size
        y1_center_px = ((image.shape[1] / tile_size) - (y1 - self.maze_space.low[1] + 1)) * tile_size
        x2_center_px = (x2 - self.maze_space.low[0] + 1) * tile_size
        y2_center_px = ((image.shape[1] / tile_size) - (y2 - self.maze_space.low[1] + 1)) * tile_size
        x_min = int(min(x1_center_px, x2_center_px))
        x_max = int(max(x1_center_px, x2_center_px))
        y_min = int(min(y1_center_px, y2_center_px))
        y_max = int(max(y1_center_px, y2_center_px))
        # Imagine a square of size width * width, with the coordinates computed above as a center. Iterate through
        # each pixel inside this square to
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                cross_product = (y2_center_px - y1_center_px) * (i - x1_center_px) \
                    - (x2_center_px - x1_center_px) * (j - y1_center_px)
                # compare versus epsilon for floating point values, or != 0 if using integers
                if abs(cross_product) > width * 10:
                    continue
                image[j, i] = color
