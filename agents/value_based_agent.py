from typing import Union
import numpy as np
import torch

from agents.agent import Agent
from abc import ABC, abstractmethod
from gym.spaces import Box, Discrete
from agents.utils.replay_buffer import ReplayBuffer


class ValueBasedAgent(Agent, ABC):

    def __init__(self, state_space: Union[Box, Discrete], action_space: Union[Box, Discrete], **params):
        """
        @param state_space: Environment's state space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """
        super().__init__(state_space, action_space, **params)

        self.batch_size = params.get("batch_size", 100)
        self.buffer_max_size = params.get("buffer_max_size", int(1e5))
        self.replay_buffer = ReplayBuffer(self.buffer_max_size, self.device)

    @abstractmethod
    def get_value(self, features, actions=None):
        return 0

    @abstractmethod
    def learn(self):
        pass

    def scale_action(self, actions: Union[np.ndarray, torch.Tensor], source_action_box: Box):
        """
        Scale an action within the given bounds action_low to action_high, to our action_space.
        The result action is also clipped to fit in the action space in case the given action wasn't exactly inside
        the given bounds.
        Useless if our action space is discrete.
        @return: scaled and clipped actions. WARNING: actions are both attribute and result. They are modified by the
        function. They are also returned for better convenience.
        """
        assert isinstance(self.action_space, Box), \
            "Scale_action is useless and cannot work if our action space is discrete."
        assert isinstance(actions, np.ndarray) or isinstance(actions, torch.Tensor)
        assert isinstance(source_action_box, Box)

        source_low, source_high = source_action_box.low, source_action_box.high
        target_low, target_high = self.action_space.low, self.action_space.high
        if isinstance(actions, torch.Tensor):
            source_low, source_high = (torch.from_numpy(source_low).to(self.device),
                                       torch.from_numpy(source_high).to(self.device))
            target_low, target_high = (torch.from_numpy(target_low).to(self.device),
                                       torch.from_numpy(target_high).to(self.device))

        # Scale action to the action space
        source_range = source_high - source_low
        target_range = target_high - target_low

        scale = target_range / source_range
        actions = actions * scale
        actions = actions + (target_low - (source_low * scale))
        clip_fun = np.clip if isinstance(actions, np.ndarray) else torch.clamp
        actions = clip_fun(actions, target_low, target_high)
        return actions

    def save_interaction(self, *interaction_data):
        """
        Function that is called to ask our agent to learn about the given interaction. This function is separated from
        self.on_action_stop(**interaction_data) because we can imagine agents that do not learn on every interaction, or
        agents that learn on interaction they didn't make (like HER that add interaction related to fake goals in their
        last trajectory).
        on_action_stop is used to update variables likes self.last_state or self.simulation_time_step_id, and
        learn_interaction is used to know the set of interactions we can learn about.

        Example: Our implementation of HER show a call to 'learn_interaction' without 'on_action_stop'
        (two last lines of 'her' file).
        """
        assert not self.under_test
        self.replay_buffer.append(interaction_data)

    def sample_training_batch(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return self.replay_buffer.sample(batch_size)

    def process_interaction(self, action, reward, new_state, done, learn=True):
        if learn and not self.under_test:
            self.save_interaction(self.last_state, action, reward, new_state, done)
            self.learn()
        super().process_interaction(action, reward, new_state, done, learn=learn)
