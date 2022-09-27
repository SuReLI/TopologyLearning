import gym
import numpy as np
from gym.spaces import Dict
import settings


class Agent:
    """
    An global agent class that describe the interactions between our agent and it's environment
    """

    def __init__(self, **params):
        self.init_params = params
        self.state_space = params.get("state_space")
        self.action_space = params.get("action_space")
        self.device = params.get("device", settings.device)
        self.name = params.get("name", "Random Agent")

        # Mandatory parameters
        assert self.state_space is not None
        assert self.action_space is not None

        if isinstance(self.state_space, Dict):
            self.state_size = self.state_space["observation"].shape[0] + self.state_space["observation"].shape[0]
            self.state_shape = self.state_space["observation"].shape
        else:
            self.state_size = self.state_space.shape[0]  # Assume observation space is continuous
            self.state_shape = self.state_space.shape
        assert len(self.state_shape) == 1

        self.continuous = isinstance(self.action_space, gym.spaces.Box)
        self.nb_actions = self.action_space.shape[0] if self.continuous else self.action_space.n
        self.last_state = None  # Useful to store interaction when we receive (new_stare, reward, done) tuple
        self.episode_id = 0
        self.episode_time_step_id = 0
        self.simulation_time_step_id = 0
        self.output_dir = None
        self.sub_plots_shape = ()

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def on_simulation_start(self):
        """
        Called when an episode is started. will be used by child class.
        """
        pass

    def on_episode_start(self, *episode_info):
        (state,) = episode_info
        assert isinstance(state, np.ndarray)
        self.last_state = state
        self.episode_time_step_id = 0

    def action(self, state):
        res = self.action_space.sample()
        return res

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        self.episode_time_step_id += 1
        self.simulation_time_step_id += 1
        self.last_state = new_state

    def on_episode_stop(self):
        self.episode_id += 1

    def on_simulation_stop(self):
        pass

    def update_plots(self, environment, sub_plots):
        """
        Update subfigures content
        """
        pass

    def reset(self):
        self.__init__(**self.init_params)

    def save(self, path):
        pass
