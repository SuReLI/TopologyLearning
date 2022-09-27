import importlib
import math
import random
from queue import PriorityQueue
from typing import Any, Tuple, Union, Dict, Optional
from gym import spaces
import numpy as np
from PIL import Image
from scipy.spatial import distance

from .utils.indexes import *
from .utils.images_fun import get_red_green_color
from utils.sys_fun import create_dir
from .maps.maps_index import MapsIndex


# un environment custom simple
def euclidean_distance(coordinates_1: tuple, coordinates_2: tuple) -> float:
    x1, y1 = coordinates_1
    x2, y2 = coordinates_2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class DiscreteGridWorld:
    start_coordinates: Tuple[Union[int, Any], int]
    metadata = {'render.modes': ['human']}

    def __init__(self, map_name:str = MapsIndex.EMPTY.value):

        self.maze_map = np.array(importlib.import_module("grid_world.environment.maps." + map_name).maze_array)
        self.height, self.width = self.maze_map.shape
        self.agent_coordinates = None
        self.state_space = spaces.Box(np.float32(-1.), np.float32(1.), (2,))
        self.action_space = spaces.Discrete(len(Direction))
        self.possible_actions = Direction

        # Window to use for human rendering mode
        self.window = None
        self.reset()

    def get_state(self, x, y):
        """
        Return a numpy array (state) that belongs to X and Y coordinates in the grid.
        """
        x_value = x + .5 - self.width / 2
        y_value = - (y + .5 - self.height / 2)
        return np.asarray([x_value, y_value])

    def get_coordinates(self, state):
        return round(state[0].item() -.5 + self.width / 2), round(- state[1].item() - .5 + self.height / 2)

    def is_valid_coordinates(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile_type(self, x, y):
        return TileType(self.maze_map[y][x].item())

    def is_terminal_tile(self, x, y):
        state_type = self.get_tile_type(x, y)
        return state_type == TileType.TERMINAL

    def is_available(self, x, y):
        # False for 218, 138
        # if we move into a row not in the grid
        if 0 > x or x >= self.width or 0 > y or y >= self.height:
            return False
        if self.get_tile_type(x, y) == TileType.WALL:
            return False
        return True

    def get_new_coordinates(self, action):
        agent_x, agent_y = self.agent_coordinates
        if Direction(action) == Direction.TOP:
            agent_y -= 1
        elif Direction(action) == Direction.BOTTOM:
            agent_y += 1
        elif Direction(action) == Direction.LEFT:
            agent_x -= 1
        elif Direction(action) == Direction.RIGHT:
            agent_x += 1
        else:
            raise AttributeError("Unknown action")
        return agent_x, agent_y

    def step(self, action):
        new_x, new_y = self.get_new_coordinates(action)
        if self.is_available(new_x, new_y):
            done = self.is_terminal_tile(new_x, new_y)
            reward = -1 if not done else 1
            self.agent_coordinates = new_x, new_y
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), reward, done
        else:
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), -1, False

    def reset(self):
        self.agent_coordinates = np.flip(random.choice(np.argwhere(self.maze_map == 2)))
        state = self.get_state(*self.agent_coordinates)
        return state

    def get_color(self, x, y, ignore_agent=False, ignore_terminals=False):
        agent_x, agent_y = self.agent_coordinates
        if (agent_x, agent_y) == (x, y) and not ignore_agent:
            return Colors.AGENT.value
        else:
            tile_type = self.get_tile_type(x, y)
            if tile_type == TileType.START:
                return Colors.START.value
            elif tile_type == TileType.WALL:
                return Colors.WALL.value
            elif tile_type == TileType.EMPTY:
                return Colors.EMPTY.value
            elif tile_type == TileType.TERMINAL:
                return Colors.EMPTY.value if ignore_terminals else Colors.TERMINAL.value
            else:
                raise AttributeError("Unknown tile type")

    def set_tile_color(self, image_array: np.ndarray, x, y, color, tile_size=10, border_size=0) -> np.ndarray:
        """
        Set a tile color with the given color in the given image as a numpy array of pixels
        :param image_array: The image where the tile should be set
        :param x: X coordinate of the tile to set
        :param y: Y coordinate of the tile to set
        :param color: new color of the tile : numpy array [Red, Green, Blue]
        :param tile_size: size of the tile in pixels
        :param border_size: size of the tile's border in pixels
        :return: The new image
        """
        tile_img = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)

        if border_size > 0:
            tile_img[:, :, :] = Colors.TILE_BORDER.value
            tile_img[border_size:-border_size, border_size:-border_size, :] = color
        else:
            tile_img[:, :, :] = color

        y_min = y * tile_size
        y_max = (y + 1) * tile_size
        x_min = x * tile_size
        x_max = (x + 1) * tile_size
        image_array[y_min:y_max, x_min:x_max, :] = tile_img
        return image_array

    def get_environment_background(self, tile_size=10, ignore_agent=True, ignore_rewards=False) -> np.ndarray:
        """
        Return an image (as a numpy array of pixels) of the environment background.
        :return: environment background -> np.ndarray
        """
        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for y in range(self.height):
            for x in range(self.width):
                cell_color = self.get_color(x, y, ignore_agent=ignore_agent, ignore_terminals=ignore_rewards)
                img = self.set_tile_color(img, x, y, cell_color)
        return img

    def get_oracle(self) -> list:
        """
        Return an oracle as a list of every possible states inside the environment.
        """
        reachable_coordinates = np.argwhere(self.maze_map != 1).tolist()
        return [self.get_state(x, y) for x, y in reachable_coordinates]

    def get_level_map(self, file_directory, file_name, oracle, values, save=True):

        high = max(values)
        low = min(values)
        distance = high - low
        if distance == 0:
            return
        for index, elt in enumerate(values):
            values[index] = (elt - low) / distance

        img = self.get_environment_background()
        for state, value in zip(oracle, values):
            color = get_red_green_color(value)
            red = int(color[1:3], 16)
            green = int(color[3:5], 16)
            blue = int(color[5:7], 16)
            x, y = self.get_coordinates(state)
            img = self.set_tile_color(img, x, y, [red, green, blue])

        if not save:
            return img

        # Save image
        image = Image.fromarray(img)
        create_dir(file_directory)
        if not file_name.endswith(".png"):
            if len(file_name.split(".")) > 1:
                file_name = "".join(file_name.split(".")[:-1])  # Remove the last extension
            file_name += ".png"
        image.save(file_directory + file_name)

    def render(self, mode='human', ignore_rewards=False):
        """
        Render the whole-grid human view
        """

        img = self.get_environment_background(ignore_agent=False, ignore_rewards=ignore_rewards)
        agent_x, agent_y = self.agent_coordinates
        self.place_point(img, self.get_state(agent_x, agent_y), Colors.AGENT.value)
        return img

    def show_skills_on_image(self, file_directory, file_name, colors, skills) -> None:
        """
        Save an image of the environment with many path draw on it
        :param file_directory: destination where the image should be saved
        :param file_name: name of the future image (without .png)
        :param colors: list of colors of shape [[R1, G1, B1], [R1, G2, B2], ... ] and len = len(paths)
        :param skills: state paths to draw, shape = [[state11, state12, ...], ...] and len = len(colors)
        """
        # Generate image
        image = self.get_environment_background()
        for path, color in zip(skills, colors):
            # Draw this path
            for state in path:
                tile_x, tile_y = self.get_coordinates(state)
                self.set_tile_color(image, tile_x, tile_y, color)

        # Save image
        image = Image.fromarray(image)
        create_dir(file_directory)
        if not file_name.endswith(".png"):
            if len(file_name.split(".")) > 1:
                file_name = "".join(file_name.split(".")[:-1])  # Remove the last extension
            file_name += ".png"
        image.save(file_directory + file_name)

    def action_space_sample(self):
        return np.random.choice(self.possible_actions)

    def get_available_positions(self, coordinates: tuple) -> list:
        """
        return an list of every available coordinates from the given one (used for A*).
        """
        x, y = coordinates  # Make sure coordinates is a tuple

        available_coordinates = []
        if x < (self.width - 1):
            new_coord = (x + 1, y)
            if self.is_available(x + 1, y):
                available_coordinates.append((new_coord, Direction.RIGHT.value))
        if x > 0:
            new_coord = (x - 1, y)
            if self.is_available(x - 1, y):
                available_coordinates.append((new_coord, Direction.LEFT.value))

        if y < (self.height - 1):
            new_coord = (x, y + 1)
            if self.is_available(x, y + 1):
                available_coordinates.append((new_coord, Direction.BOTTOM.value))
        if y > 0:
            new_coord = (x, y - 1)
            if self.is_available(x, y - 1):
                available_coordinates.append((new_coord, Direction.TOP.value))

        return available_coordinates

    def best_path(self, state_1, state_2):
        """
        Return the shortest distance between two tiles, in number of action the agent needs to go from one to another.
        :param state_1: Start state,
        :param state_2: Destination state,
        :return: Shortest path (using A*), as a list of action to move from state_1 to state_2.
        """
        # Remove trivial case
        if isinstance(state_1, tuple):
            coordinates_1 = state_1
        else:
            coordinates_1 = self.get_coordinates(state_1)
        if isinstance(state_2, tuple):
            coordinates_2 = state_2
        else:
            coordinates_2 = self.get_coordinates(state_2)

        frontier = PriorityQueue()
        frontier.put((0, coordinates_1))
        came_from: Dict[tuple, Optional[tuple]] = {}
        cost_so_far = {coordinates_1: 0}
        came_from[coordinates_1] = None

        while not frontier.empty():
            priority, current = frontier.get()

            if current == coordinates_2:
                break

            for next_position, action in self.get_available_positions(current):
                new_cost = cost_so_far[current] + 1
                if next_position not in cost_so_far or new_cost < cost_so_far[next_position]:
                    cost_so_far[next_position] = new_cost
                    priority = new_cost + euclidean_distance(next_position, coordinates_2)
                    frontier.put((priority, next_position))
                    came_from[next_position] = current

        return came_from, cost_so_far

    def distance(self, state_1, state_2):
        _, distance = self.best_path(state_1, state_2)
        coordinates_2 = self.get_coordinates(state_2)
        return distance[coordinates_2]

    def show_path_on_image(self, state_1, state_2, file_directory, file_name, colors) -> None:
        """
        Save an image of the environment with many path draw on it
        :param state_1: start state,
        :param state_2: destination state,
        :param file_directory: destination where the image should be saved
        :param file_name: name of the future image (without .png)
        :param colors: list of colors of shape [[R1, G1, B1], [R1, G2, B2], ... ] and len = len(paths)
        """

        best_path = self.best_path(state_1, state_2)

        if isinstance(state_1, tuple):
            coordinates_1 = state_1
        else:
            coordinates_1 = self.get_coordinates(state_1)
        if isinstance(state_2, tuple):
            coordinates_2 = state_2
        else:
            coordinates_2 = self.get_coordinates(state_2)

        if not self.is_available(coordinates_1[0], coordinates_1[1]) or not self.is_available(coordinates_2[0],
                                                                                              coordinates_2[1]):
            print("one of these states is not available")

        # Generate image
        image = self.get_environment_background()

        # Draw this path
        for coordinates in best_path.coordinates:
            tile_x, tile_y = coordinates
            self.set_tile_color(image, tile_x, tile_y, colors)

        # Save image
        image = Image.fromarray(image)
        create_dir(file_directory)
        if not file_name.endswith(".png"):
            if len(file_name.split(".")) > 1:
                file_name = "".join(file_name.split(".")[:-1])  # Remove the last extension
            file_name += ".png"
        image.save(file_directory + file_name)

    def place_point(self, image: np.ndarray, state, color: np.ndarray, width=5):
        """
        Modify the input image
        param image: Initial image that will be modified.
        param x: x coordinate in the state space of the point to place.
        param y: y coordinate in the state space of the point to place.
        param color: Color to give to the pixels that compose the point.
        param width: Width of the circle (in pixels).
        """


        center_x, center_y = self.get_coordinates(state)
        center_x /= self.width
        center_y /= self.height
        center_y, center_x = (image.shape[:2] * np.array([center_y, center_x])).astype(int)

        # Imagine a square of size width * width, with the coordinates computed above as a center. Iterate through
        # each pixel inside this square to
        radius = round(width / 2) + 1
        for i in range(center_x - radius, center_x + radius):
            for j in range(center_y - radius, center_y + radius):
                if distance.euclidean((i + 0.5, j + 0.5), (center_x, center_y)) < radius:
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

        center_x_1, center_y_1 = self.get_coordinates(state_1)
        center_x_1 /= self.width
        center_y_1 /= self.height
        center_y_1, center_x_1 = (image.shape[:2] * np.array([center_y_1, center_x_1])).astype(int)

        center_x_2, center_y_2 = self.get_coordinates(state_2)
        center_x_2 /= self.width
        center_y_2 /= self.height
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
