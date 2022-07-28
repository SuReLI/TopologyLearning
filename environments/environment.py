import abc
from typing import Tuple

import numpy as np


class Environment(abc.ABC):

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment
        :return: a np.ndarray of a new state.
        """
        pass

    @abc.abstractmethod
    def step(self, action) -> Tuple[np.ndarrray, float, bool]:
        """
        Perform the action in the environment.
        :return: a tuple of a new state (np.ndarray), a reward (float), and a boolean that indicate if the episode is
        done or not.
        """
        pass

    @abc.abstractmethod
    def render(self) -> np.ndarray:
        """
        Render the environment.
        :return: This function should return a np.ndarray in pixel, that can be converted in image after.
        """
        pass


class GoalReachingEnvironment(Environment):
    def __init__(self, observation_space, action_space, goal_size):
        self.observation_space = observation_space
        self.action_space = action_space
        self.goal_size = goal_size  # Goal size is mandatory to make sure the agent can build its neural networks.

    @abc.abstractmethod
    def reset(self) -> Tuple[np.ndarrray, np.ndarray]:
        """
        Reset the environment
        :return: a tuple of a new state and a new goal
        """
        pass

    @abc.abstractmethod
    def extract_goal(self, state) -> np.ndarrray:
        """
        Convert the given state into a goal.
        :return: A goal as type np.ndarray.
        """
        pass


