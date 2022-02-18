import gym
import torch
from settings import settings


class Agent:
    """
    An global agent class that describe the interactions between our agent and it's environment
    """

    def __init__(self, state_space, action_space, device=settings.device, name="Random Agent"):
        self.name = name  # The name is used inside plot legend, outputs directory path, and outputs file names

        self.state_space = state_space
        self.state_shape = state_space.shape
        self.state_size = state_space.shape[0]  # Assume state space is continuous
        assert len(self.state_shape) == 1

        self.continuous = isinstance(action_space, gym.spaces.Box)
        self.action_space = action_space
        self.nb_actions = self.action_space.shape[0] if self.continuous else self.action_space.n
        self.last_state = None  # Useful to store interaction when we receive (new_stare, reward, done) tuple
        self.device = device
        self.episode_id = 0
        self.episode_time_step_id = 0
        self.time_step_id = 0

    def on_simulation_start(self):
        """
        Called when an episode is started. will be used by child class.
        """
        pass

    def on_episode_start(self, *episode_info):
        (state, episode_id) = episode_info
        self.last_state = state
        self.episode_time_step_id = 0
        self.episode_id = episode_id

    def action(self, state):
        res = self.action_space.sample()
        return res

    def on_action_stop(self, action, new_state, reward, done):
        self.episode_time_step_id += 1
        self.time_step_id += 1
        self.last_state = new_state

    def on_episode_stop(self):
        self.episode_id += 1

    def on_simulation_stop(self):
        pass

    # Plot functions
    def initiate_subplots(self, outer_figure):
        """
        Initialise subfigures using a given outer figure place.
        """
        pass

    def update_plots(self, environment):
        """
        Update subfigures content
        """
        pass

    def reset(self):
        self.__init__(self.state_space, self.action_space, self.device, self.name)
