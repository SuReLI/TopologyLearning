import copy
import json
from typing import Union

import gym
from gym.spaces import Box, Discrete
import numpy as np
import torch


class Agent:
    """
    A global agent class that describe the interactions between our agent, and it's environment.
    """

    name = "Agent"

    def __init__(self, state_space: Union[Box, Discrete], action_space: Union[Box, Discrete], **params):
        """
        @param state_space: Environment's state space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """
        assert isinstance(state_space, Box) or isinstance(state_space, Discrete), \
            "The state space should be an instance of gym.spaces.Space"
        assert isinstance(action_space, Box) or isinstance(action_space, Discrete), \
            "The action space should be an instance of gym.spaces.Space"

        self.init_params = params
        self.state_space = state_space
        self.action_space = action_space
        self.device = params.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Mandatory parameters
        assert not isinstance(self.state_space, dict), "state space as dictionary is not supported."
        self.state_size = self.state_space.shape[0]  # Assume observation space is continuous
        self.state_shape = self.state_space.shape

        self.continuous = isinstance(self.action_space, gym.spaces.Box)
        self.nb_actions = self.action_space.shape[0] if self.continuous else self.action_space.n
        self.last_state = None  # Useful to store interaction when we receive (new_stare, reward, done) tuple
        self.episode_id = 0
        self.episode_time_step_id = 0
        self.simulation_time_step_id = 0
        self.output_dir = None
        self.sub_plots_shape = ()
        self.under_test = False
        self.episode_started = False

    def start_episode(self, state: np.ndarray, test_episode=False):
        if self.episode_started:
            self.stop_episode()
        self.episode_started = True
        self.last_state = state
        self.episode_time_step_id = 0
        self.under_test = test_episode

    def action(self, state, explore=True):
        return self.action_space.sample()

    def process_interaction(self, action, reward, new_state, done, learn=True):
        self.episode_time_step_id += 1
        self.simulation_time_step_id += 1
        self.last_state = new_state

    def stop_episode(self):
        self.episode_id += 1
        self.episode_started = False

    def reset(self):
        self.__init__(self.state_space, self.action_space, **self.init_params)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, Agent):
                if v.name == "AntMazeControlPolicy":
                    a = 1

                setattr(result, k, v.copy())
            elif isinstance(v, dict):
                new_dict = {}
                for k_, v_ in v.items():
                    new_dict[k_] = v_.copy() if k_ == "goal_reaching_agent" else copy.deepcopy(v_)
                setattr(result, k, new_dict)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def copy(self):
        return copy.deepcopy(self)
