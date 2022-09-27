"""
GWR is a topological graph building strategy, that build a graph that fit the topology of a dataset.
https://www.sciencedirect.com/science/article/pii/S0893608002000783
"""
import copy
import random
import numpy as np
import torch

from agents.graph_planning.rgl import RGL
from agents.graph_planning.topological_graph_planning_agent import PlanningTopologyLearner, TopologyLearnerMode
from agents.utils.mlp import MLP
from torch import optim
from torch.nn import ReLU, Sigmoid


class STC(RGL):
    def __init__(self, **params):
        params["re_usable_policy"] = False
        params["name"] = params.get("name", "STC")
        super(STC, self).__init__(**params)
        self.ti_tc_network = params.get("translation_invariant_tc_network", False)
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
        self.edges_similarity_threshold = params.get("edges_similarity_threshold", 0.5)
        self.nodes_similarity_threshold = params.get("nodes_similarity_threshold", 0.6)

        self.tc_errors_memory = []
        self.tc_average_errors_memory = []
        self.label_0_values = {}
        self.label_1_values = {}

        super().__init__(**params)

    def on_pre_training_done(self, start_state, reached_goals):
        """
        Compute the longer distance estimation over every goal that has been reached during the pre-training.
        It allows to choose reachability parameters more easily.
        """
        pass

    def extend_graph(self):

        assert self.last_exploration_trajectory != []

        for state in self.last_exploration_trajectory:
            nodes_to_link = []
            for node_id, node_parameters in self.topology.nodes(data=True):
                node_state = node_parameters["state"]  # Position of node in the observation space
                estimated_similarity = self.get_similarity(node_state, state)
                if estimated_similarity >= self.nodes_similarity_threshold:
                    break
                if estimated_similarity >= self.edges_distance_threshold:
                    nodes_to_link.append(node_id)
            else:
                # => this observation is far enough from any nodes
                if nodes_to_link:  # Prevent to create unliked nodes
                    # Create node
                    new_node = self.create_node(state)

                    # Create edges
                    if self.last_node_explored not in nodes_to_link:
                        nodes_to_link.append(self.last_node_explored)
                    for node_id in nodes_to_link:
                        self.create_edge(new_node, node_id, cost=1)

    def get_similarity(self, state_1, state_2):
        """
        Return a boolean reflecting the similarity of two states according to the TC network.
        """
        state_1 = torch.from_numpy(state_1).to(self.device)
        state_2 = torch.from_numpy(state_2).to(self.device)
        if self.ti_tc_network:
            input_ = torch.concat((state_1[:self.goal_size] - state_2[:self.goal_size], state_1[self.goal_size:]), dim=-1)
        else:
            input_ = torch.concat((state_1, state_2), dim=-1)
        with torch.no_grad():
            return self.topological_cognition_network(input_).item()

    def get_distance_estimation(self, state, goal):
        """
        Use the UVFA to get a value function approximation between two states.
        """
        return - self.get_similarity(state, goal)

    def on_episode_stop(self, learn=None):
        if learn is None:
            learn = self.mode == TopologyLearnerMode.GO_TO
        super().on_episode_stop(learn)

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
            label = 0 if distance > self.targeted_edge_length else 1
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
                inputs = torch.concat((states_1[:, :self.goal_size] - states_2[:, :self.goal_size], states_1[:, self.goal_size:]), dim=-1)
            else:
                inputs = torch.concat((states_1, states_2), dim=-1)

            # Predict label and learn loss
            predictions = self.topological_cognition_network(inputs).squeeze()
            labels = torch.Tensor(list(labels)).to(device=self.device, dtype=torch.float32)
            error = self.tc_criterion(predictions, labels)

            # self.topological_cognition_network.optimizer.zero_grad()
            # error.backward()
            # self.topological_cognition_network.optimizer.learning_step()

            self.topological_cognition_network.learn(error)
