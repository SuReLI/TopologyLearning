from random import choice
from typing import Tuple
from ..environment import Environment
from maps.map_reader import read_maze_file, TileType
import numpy as np
import gym
from gym.spaces import Box


class PointEnv(Environment):
    """Abstract class for 2D navigation environments."""

    def __init__(self, maze_name, action_noise=1.0, episode_duration=100, **rendering_options):
        """
        Initialize the point maze environment.
        :param maze_name: name of the maze. It should be precise enough to find the maze specification file in maps/
        directory. In other words, the maze spec file "maps/" + maze_name + ".txt" will be used to build the maze.
        :param action_noise: define the std of the gaussian noise added to the action.
        """
        self.maze = read_maze_file(maze_name)
        self.action_noise = action_noise
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32)
        self.observation_space = Box(
            low=np.array([0.0, 0.0]),
            high=np.array([self.maze.height, self.maze.width]),
            dtype=np.float32)
        self.state = None  # will be set on reset().
        self.episode_duration = 100
        self.current_episode_step = 0
        self.reset()

        # Rendering options
        self.px_per_tile = rendering_options.get("px_per_tile", 10)
        self.empty_tile_color = rendering_options.get("empty_tile_color", [255, 255, 255])
        self.wall_color = rendering_options.get("wall_color", [0, 0, 0])
        self.start_color = rendering_options.get("start_color", [0, 0, 255])
        self.target_color = rendering_options.get("target_color", [0, 255, 0])
        self.agent_color = rendering_options.get("target_color", [255, 0, 0])
        self.agent_width_in_px = rendering_options.get("agent_width_in_px", 5)

    def _sample_empty_state(self):
        new_state = np.array(choice(self.maze.empty_states), dtype=np.float)
        new_state += np.random.uniform(size=2)
        assert not self._is_blocked(new_state)
        return new_state

    def _normalize_obs(self, obs):
        return np.array([
            obs[0] / float(self.maze.width),
            obs[1] / float(self.maze.height)
        ])

    def reset(self) -> np.ndarray:
        self.current_episode_step = 0
        self.state = self._sample_empty_state()
        return self._normalize_obs(self.state.copy())

    def _discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int)

        # Round down to the nearest cell if at the boundary.
        if i == self.maze.width:
            i -= 1
        if j == self.maze.height:
            j -= 1
        return i, j

    def _is_blocked(self, state):
        if not self.observation_space.contains(state):
            return True
        (i, j) = self._discretize_state(state)
        return self.maze.map[i, j] == 1

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        assert self.current_episode_step == self.episode_duration, \
            "Environment hasn't been reset after an episode ends."
        if self.action_noise > 0:
            action += np.random.normal(0, self.action_noise)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)
        num_sub_steps = 10
        dt = 1.0 / num_sub_steps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_sub_steps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state
        self.current_episode_step += 1
        done = self.current_episode_step == self.episode_duration
        rew = 0. if tuple(self.state) in self.maze.targets_coordinates else -1.
        return self._normalize_obs(self.state.copy()), rew, done

    def color_tile(self, image, x, y, width, color) -> np.ndarray:
        tile_img = np.zeros(shape=(width, width, 3), dtype=np.uint8)
        tile_img[:, :, :] = color

        y_min = int(y * width)
        y_max = int((y + 1) * width)
        x_min = int(x * width)
        x_max = int((x + 1) * width)
        image[y_min:y_max, x_min:x_max, :] = tile_img
        return image

    def get_background_image(self) -> np.ndarray:
        image = np.zeros((self.maze.width * self.px_per_tile, self.maze.height * self.px_per_tile, 3))
        for row_id, row in enumerate(self.maze.map):
            for col_id, tile in enumerate(row):
                if tile == TileType.EMPTY:
                    color = self.empty_tile_color
                elif tile == TileType.WALL:
                    color = self.wall_color
                elif tile == TileType.START:
                    color = self.start_color
                else:
                    color = self.target_color

                # Color a tile with the color and place it in the image.
                image = self.color_tile(image, col_id, row_id, self.px_per_tile, color)
        return image

    def render(self) -> np.ndarray:

        # Place the agent on image
        image = self.color_tile(image, *self.state, self.agent_width_in_px, self.agent_color)
        return image

