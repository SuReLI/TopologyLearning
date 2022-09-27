import importlib
import math
import random
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Box
from .maps.maps_index import MapsIndex
from point_maze.environment.maps.tile_type import TileType

class Colors(Enum):
    EMPTY = [250, 250, 250]
    WALL = [50, 54, 51]
    START = [213, 219, 214]
    TERMINAL = [73, 179, 101]
    TILE_BORDER = [50, 54, 51]
    AGENT = [219, 0, 0]
    GOAL = [255, 0, 0]

class PointMaze:
    """
    A Point Maze environment.
        The agent is a point inside a maze and interact with the environment by choosing a movement velocity (< 0 to go
    backward, and > 0 to go forward) and a rotation velocity (< 0 to go right, and > 0 to go left).
        These values are between -1 and 1, and are multiplied by a value that correspond to the maximum speed in both
    actions. Note that a too high value as the maximum movement velocity available may allow the agent to cross walls.
        A noise is added to the action. This is a gaussian noise with mean 0 and std noise std.
    """

    def __init__(self, map_name=MapsIndex.EMPTY.value, noise_std=np.array([0.1, 0.1])):
        """
        Initialize the point environment.
        :param str map_name: The name of the map to load. Should correspond to a file name in ./maps.
        :param np.ndarray noise_std: The standard deviation of the nose applied to the action.
        """
        self.maze_map = np.array(importlib.import_module("point_maze.environment.maps." + map_name).maze_array)
        self.height, self.width = self.maze_map.shape
        self.action_noise_std = noise_std

        # Store higher action allowed
        self.higher_action = np.array([0.4, 0.8])
        assert (self.higher_action <= np.array([5, math.pi - 1e-6])).all()

        self.state = None
        self.action_space = Box(-1, 1, (2,))
        assert noise_std.shape == self.action_space.low.shape
        self.state_space = Box(low=np.array([0, 0, -math.pi]), high=np.array([self.width, self.height, math.pi]))
        self.reset()

        self.render_tile_size = 10

    def reset(self):
        # Choose a start location

        self.state = np.append(np.flip(random.choice(np.argwhere(self.maze_map == TileType.START.value))),
                               0,).astype(float)
        # coordinates + angle
        self.state[0] += 0.5 - self.width / 2
        self.state[1] = - (self.state[1] + 0.5 - self.height / 2)
        return self.state.copy()

    def get_action_noise(self):
        return np.random.normal(0, self.action_noise_std, (2,))

    def step(self, action):
        if (self.action_noise_std > 0).all():
            action = np.multiply(action, self.higher_action)
            action += self.get_action_noise()
        action = np.clip(action, np.multiply(self.higher_action, self.action_space.low),
                         np.multiply(self.higher_action, self.action_space.high))
        num_sub_steps = 10
        action_part = action / num_sub_steps
        for _ in range(num_sub_steps):
            new_state = self.state.copy()

            # Operate rotation
            new_angle = new_state[-1].item() + action_part[-1]
            if new_angle < 0:
                new_angle = math.pi + new_angle
            elif new_angle > math.pi:
                new_angle %= math.pi
            new_state[-1] = new_angle

            # Operate speed action
            new_x = new_state[0].item() + action_part[0].item() * math.cos(new_angle)
            new_y = new_state[1].item() + action_part[0].item() * math.sin(new_angle)
            new_tile = self.get_tile(np.array([new_x, new_y]))
            if new_tile is not None and new_tile != 1:
                new_state[0] = new_x
                new_state[1] = new_y
            self.state = new_state.copy()
        reward = 1. if self.get_tile(self.state[:2]) == TileType.TERMINAL else 0.
        return self.state.copy(), reward, False

    def get_tile_coordinates(self, state):
        """
        Return the coordinates (y, x) of the tile where the state with coordinate (x, y) is inside.
        Note that we consider that the center of a tile with coordinates a, b is a + 0.5, b + 0.5; Then, a and b
        coordinates correspond to the bottom-left corner of the tile.
        :return: the tile coordinates
        """
        tile_x = int(state[0].item() + self.width / 2)
        tile_y = int(- state[1].item() + self.height / 2)
        # '-> because If y is high, then we are at the top of the maze, so the index of the tile is low (first row of
        # the array). Change this system implies to change the way the next position is computed using the angle.
        return tile_y, tile_x

    def get_tile(self, state):
        tile_y, tile_x = self.get_tile_coordinates(state)
        return self.maze_map[tile_y, tile_x].item()

    def render(self, with_agent=True, ignore_rewards=False):
        """
        Return a numpy array that correspond to a rgb image of the environment.
        :param bool with_agent: Place a character (point + line, indicating the direction) that show the agent state at
        the time the images is generated.
        :param ignore_rewards: Don't show rewarding states if True, show them otherwise.
        :return np.ndarray: The result rgb image
        """
        image = np.zeros((self.height * self.render_tile_size, self.width * self.render_tile_size, 3)).astype(np.uint8)

        for i in range(self.maze_map.shape[0]):
            for j in range(self.maze_map.shape[1]):
                color = Colors.EMPTY.value
                if self.maze_map[i, j] == TileType.WALL.value:
                    color = Colors.WALL.value
                if not ignore_rewards and self.maze_map[i, j] == TileType.TERMINAL.value:
                    color = Colors.TERMINAL.value
                image[i * self.render_tile_size: (i + 1) * self.render_tile_size,
                    j * self.render_tile_size: (j + 1) * self.render_tile_size] = color

        if with_agent:
            self.place_point(image, self.state, color=Colors.AGENT.value)

        return image

    def place_point(self, image, state, color, width=5):
        # Draw a circle of size tile_size / 10 on agent's position
        x = int((state[0].item() + self.width / 2) * self.render_tile_size)
        y = int((- state[1].item() + self.height / 2) * self.render_tile_size)
        circle_radius = int(width / 2 + 1)

        # Explanations of the general idea are here:
        # https://stackoverflow.com/questions/10031580/how-to-write-simple-geometric-shapes-into-numpy-arrays
        drawing_area_start_x = int(y - (circle_radius + 1))  # position - circle radius, with some margin
        drawing_area_start_y = int(x - (circle_radius + 1))  # to make sure every pixel is drawn
        drawing_area_size = circle_radius * 2 + 2
        xx, yy = np.mgrid[drawing_area_start_x:drawing_area_start_x + drawing_area_size,
                 drawing_area_start_y:drawing_area_start_y + drawing_area_size]
        circle = (xx - x) ** 2 + (yy - y) ** 2
        circle_filter = circle < circle_radius ** 2
        image[drawing_area_start_x:drawing_area_start_x + drawing_area_size,
            drawing_area_start_y:drawing_area_start_y + drawing_area_size][circle_filter] = color

    def place_edge(self, image: np.ndarray, state_1, state_2, color: np.ndarray, width=5):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the state space of the point to place.
        param y: y coordinate in the state space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """

        center_x_1 = int(state_1[0].item() + self.width / 2) / self.width
        center_y_1 = int(- state_1[1].item() + self.height / 2) / self.height
        center_y_1, center_x_1 = (image.shape[:2] * np.array([center_y_1, center_x_1])).astype(int)

        center_x_2 = int(state_2[0].item() + self.width / 2) / self.width
        center_y_2 = int(- state_2[1].item() + self.height / 2) / self.height
        center_y_2, center_x_2 = (image.shape[:2] * np.array([center_y_2, center_x_2])).astype(int)

        # Convert x, y state_space_position into maze_coordinates to get point center
        # Explanation: same as in self.set_space_color, lines of code 5/6
        x_min = int(min(center_x_1, center_x_2))
        x_max = int(max(center_x_1, center_x_2))
        y_min = int(min(center_y_1, center_y_2))
        y_max = int(max(center_y_1, center_y_2))
        # Imagine a square of size width * width, with the coordinates computed above as a center. Iterate through
        # each pixel inside this square to
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                cross_product = (center_y_2 - center_y_1) * (i - center_x_1) \
                    - (center_x_2 - center_x_1) * (j - center_y_1)
                # compare versus epsilon for floating point values, or != 0 if using integers
                if abs(cross_product) > width * 10:
                    continue
                image[j, i] = color
