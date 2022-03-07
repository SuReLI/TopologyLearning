# Goal conditioned agent

import gym
import numpy as np
import torch
from settings import settings
from agents.agent import Agent


class GoalConditionedAgent(Agent):
    """
    An global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    def __init__(self, state_space, action_space, device=settings.device, name="Random agent"):
        super().__init__(state_space, action_space, device, name=name)
        self.current_goal = None

    def on_episode_start(self, *args):
        state, goal = args
        assert isinstance(state, np.ndarray)
        assert isinstance(goal, np.ndarray)
        super().on_episode_start(state)
        self.current_goal = goal
