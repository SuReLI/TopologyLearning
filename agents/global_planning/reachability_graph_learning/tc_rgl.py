"""
GWR is a topological graph building strategy, that build a graph that fit the reachability graph of a dataset.
https://www.sciencedirect.com/science/article/pii/S0893608002000783
"""
import random
from typing import Union

import numpy as np
import torch
from gym.spaces import Box, Discrete
from agents.global_planning.reachability_graph_learning.rgl import RGL
from agents.utils.nn import MLP
from torch import optim
from torch.nn import ReLU, Sigmoid


class TC_RGL(RGL):

    name = "TC-RGL"

    def __init__(self, goal_conditioned_wrapper, value_based_agent_class,
                 state_space: Union[Box, Discrete], action_space: Union[Box, Discrete], **params):
        """
        @param goal_conditioned_wrapper: Wrapper used to turn the given value based agent class into a goal
            conditioned agent.
        @param value_based_agent_class: Value based agent clas
        @param state_space: Environment's state space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        params["re_usable_policy"] = True
        super().__init__(goal_conditioned_wrapper, value_based_agent_class, state_space, action_space, **params)
        self.ti_tc_network = params.get("translation_invariant_tc_network", True)
        assert isinstance(self.ti_tc_network, bool)
        self.tc_layer_1_size = params.get("tc_layer_1_size", 125)
        self.tc_layer_2_size = params.get("tc_layer_2_size", 100)
        self.tc_learning_rate = params.get("tc_learning_rate", 0.001)
        self.tc_batch_size = params.get("tc_batch_size", 250)
        self.tc_buffer_max_size = params.get("tc_buffer_max_size", 1e9)
        self.nb_tc_data_seen = 0
        self.tc_criterion = params.get("tc_criterion", torch.nn.MSELoss())

        self.targeted_edge_length = params.get("targeted_edge_length", 20)  # Hyperparameter k in stc paper

        self.topological_cognition_network = MLP(self.state_size if self.ti_tc_network else self.state_size * 2,
                                                 self.tc_layer_1_size, ReLU(),
                                                 self.tc_layer_2_size, ReLU(),
                                                 1, Sigmoid(),
                                                 learning_rate=self.tc_learning_rate, optimizer_class=optim.Adam,
                                                 device=self.device).float()

        # We use to replay buffers to have the same amount of positive / negative data in a batch, to limit over-fitting
        self.tc_replay_buffer_positive = []  # Replay buffer for samples with label = 1
        self.tc_replay_buffer_positive = []  #                            ... label = 0
        self.tc_replay_buffer_negative = []
        self.last_episode_trajectory = []

        self.tc_errors_memory = []
        self.tc_average_errors_memory = []
        self.label_0_values = {}
        self.label_1_values = {}

        super().__init__(goal_conditioned_wrapper, value_based_agent_class, state_space, action_space, **params)

    def on_pre_training_done(self, start_state, reached_goals):
        """
        Compute the longer distance estimation over every goal that has been reached during the pre-training.
        It allows to choose reachability parameters more easily.
        """
        pass

    def get_distance_estimation(self, states_batch, goals_batch):
        """
        Return a boolean reflecting the distance of two states according to the TC network.
        """
        if isinstance(states_batch, np.ndarray):
            states_batch = torch.from_numpy(states_batch).to(self.device)
        if isinstance(goals_batch, np.ndarray):
            goals_batch = torch.from_numpy(goals_batch).to(self.device)
        if self.ti_tc_network:
            input_ = torch.concat((states_batch[:, :self.goal_size] - goals_batch[:, :self.goal_size],
                                   states_batch[:, self.goal_size:]), dim=-1)
        else:
            input_ = torch.concat((states_batch, states_batch), dim=-1)
        with torch.no_grad():
            return self.topological_cognition_network(input_).cpu().detach().numpy()

    def get_normalised_distance_estimation(self, states, goals):
        """
        Use the UVFA to get a value function approximation between two states.
        """

        # Convert inputs to batch
        if len(states.shape) == 1:
            states = states[np.newaxis]
        if len(goals.shape) == 1:
            goals = goals[np.newaxis]

        # Operate projections: States should be in the state space and goals in the sub-goal space.
        # - Inverse projection G -> S if the states don't have the right size
        #   (computing distance between goal and state).
        if states.shape[-1] != self.state_size:
            if states.shape[-1] != self.goal_size:
                raise ValueError("Unknown state shape.")
            states = self.goal_to_state(states)

        # - Projection S -> G if the given sub-goals are actually states or goals. Our control policy only know
        #   goals, not goals.
        if goals.shape[-1] != self.goal_size:
            if goals.shape != self.state_size:
                raise ValueError("Unknown goal shape")
            goals = goals[self.state_to_goal_filter]

        # Now inputs have the right size. We can compute the distance.
        # Return the normalised estimated distance (assuming self.distance_estimation_min == 0)
        return self.get_distance_estimation(states, goals).flatten()

    def store_tc_training_samples(self, last_trajectory):
        """
        Use self.last_episode_trajectory to generate and store training samples for the TC network
        """

        for sample_id in range(len(last_trajectory) // 2):

            # Compute data index in the buffer using reservoir sampling
            state_1_index = random.randint(0, len(last_trajectory) - 1)
            state_2_index = random.randint(0, len(last_trajectory) - 1)
            distance = abs(state_2_index - state_1_index)
            state_1 = last_trajectory[state_1_index]
            state_2 = last_trajectory[state_2_index]
            label = 0 if distance < self.targeted_edge_length else 1
            if label == 0:
                if len(self.tc_replay_buffer_negative) > self.tc_buffer_max_size // 2:
                    self.tc_replay_buffer_negative.pop(0)
                self.tc_replay_buffer_negative.append((state_1, state_2, label))
            elif label == 1:
                if len(self.tc_replay_buffer_positive) > self.tc_buffer_max_size // 2:
                    self.tc_replay_buffer_positive.pop(0)
                self.tc_replay_buffer_positive.append((state_1, state_2, label))

        self.train_tc_network()

    def train_tc_network(self):
        if len(self.tc_replay_buffer_positive) > self.tc_batch_size // 2 \
                and len(self.tc_replay_buffer_negative) > self.tc_batch_size // 2:

            # Sample batch data
            p_states_1, p_states_2, p_labels = \
                list(zip(*random.sample(self.tc_replay_buffer_positive, self.tc_batch_size // 2)))
            n_states_1, n_states_2, n_labels = \
                list(zip(*random.sample(self.tc_replay_buffer_negative, self.tc_batch_size // 2)))
            states_1 = p_states_1 + n_states_1
            states_2 = p_states_2 + n_states_2
            labels = p_labels + n_labels
            states_1 = torch.from_numpy(np.array(states_1)).to(self.device)
            states_2 = torch.from_numpy(np.array(states_2)).to(self.device)
            if self.ti_tc_network:
                inputs = torch.concat((states_1[:, :self.goal_size] - states_2[:, :self.goal_size],
                                       states_1[:, self.goal_size:]), dim=-1)
            else:
                inputs = torch.concat((states_1, states_2), dim=-1)

            # Predict label and learn loss
            predictions = self.topological_cognition_network(inputs).squeeze()
            labels = torch.Tensor(list(labels)).to(device=self.device, dtype=torch.float32)
            error = self.tc_criterion(predictions, labels)

            self.topological_cognition_network.learn(error)
