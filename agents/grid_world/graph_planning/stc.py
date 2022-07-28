"""
GWR is a topological graph building strategy, that build a graph that fit the topology of a dataset.
https://www.sciencedirect.com/science/article/pii/S0893608002000783
"""
import random
import numpy as np
import torch

from old.src.agents.grid_world.graph_planning.topological_graph_planning_agent import PlanningTopologyLearner, TopologyLearnerMode
from old.src.agents.utils.mlp import MLP
from torch import optim
from torch.nn import ReLU, Sigmoid


class STC_TL(PlanningTopologyLearner):
    def __init__(self, **params):

        super(STC_TL, self).__init__(**params)

        self.tc_layer_1_size = params.get("tc_layer_1_size", 125)
        self.tc_layer_2_size = params.get("tc_layer_2_size", 100)
        self.tc_learning_rate = params.get("tc_learning_rate", 0.001)
        self.tc_batch_size = params.get("tc_batch_size", 125)
        self.tc_buffer_max_size = params.get("tc_buffer_max_size", 1e9)
        self.nb_tc_data_seen = 0
        self.tc_batch_size = params.get("tc_batch_size", 125)
        self.tc_criterion = params.get("tc_criterion", torch.nn.MSELoss())

        self.targeted_edge_length = params.get("targeted_edge_length", 10)  # Hyperparameter k in stc paper

        self.topological_cognition_network = MLP(4, self.tc_layer_1_size, ReLU(),
                                                 self.tc_layer_2_size, ReLU(), 1, Sigmoid(),
                                                 learning_rate=self.tc_learning_rate, optimizer_class=optim.Adam,
                                                 device=self.device).float()
        # We use to replay buffers to have the same amount of positive / negative data in a batch, to limit over-fitting
        self.tc_replay_buffer_positive = []  # Replay buffer for samples with label = 1
        self.tc_replay_buffer_positive = []  #                            ... label = 0
        self.tc_replay_buffer_negative = []
        self.last_episode_trajectory = []
        self.tc_reachability_threshold = params.get("tc_reachability_threshold", 0.5)
        self.tc_node_creation_threshold = params.get("tc_reachability_threshold", 0.6)

        self.tc_errors_memory = []
        self.tc_average_errors_memory = []
        self.label_0_values = {}
        self.label_1_values = {}

        params["name"] = "STC"
        super().__init__(**params)

    def on_episode_start(self, *args):
        state = args[0]
        self.last_episode_trajectory = [state]
        super().on_episode_start(*args)

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        self.last_episode_trajectory.append(new_state)
        self.train_tc_network()

        super().on_action_stop(action, new_state, reward, done)

    def extend_graph(self):
        assert self.last_exploration_trajectory != []

        for state in self.last_exploration_trajectory:
            # Iterate through every node to search for one that cover the given observation:
            node_creation = False
            for node_id, node_parameters in self.topology.nodes(data=True):
                node_state = node_parameters["state"]
                # Compute the similarity between the node and the observation
                similarity = self.get_similarity(node_state, state)
                if self.tc_node_creation_threshold < similarity:
                    break  # This observation is already covered by another node
                elif self.tc_reachability_threshold < similarity:
                    node_creation = True
            else:  # If no break has been called (aka. the observation isn't covered)
                if node_creation:
                    new_node_id = self.create_node(state)  # Make sure the node is well created

                    # Create edges
                    for node_id, node_parameters in self.topology.nodes(data=True):
                        node_state = node_parameters["state"]

                        if self.tc_reachability_threshold < self.get_similarity(node_state, state) \
                                < self.tc_node_creation_threshold:
                            self.create_edge(node_id, new_node_id)

    def get_similarity(self, state_1, state_2):
        """
        Return a boolean reflecting the similarity of two states according to the TC network.
        """
        state_1 = torch.from_numpy(state_1).to(self.device)
        state_2 = torch.from_numpy(state_2).to(self.device)
        input_ = torch.concat((state_1, state_2), dim=-1)
        with torch.no_grad():
            return self.topological_cognition_network(input_).item()

    def on_episode_stop(self, learn=None):
        if learn is None:
            learn = self.mode == TopologyLearnerMode.GO_TO
        if learn:
            self.store_tc_training_samples()
        super().on_episode_stop(learn)

    def store_tc_training_samples(self):
        """
        Use self.last_episode_trajectory to generate and store training samples for the TC network
        """

        for sample_id in range(len(self.last_episode_trajectory) // 2):

            # Compute data index in the buffer using reservoir sampling
            """
            index = random.randint(0, self.nb_tc_data_seen)
            self.nb_tc_data_seen += 1
            if index >= self.tc_buffer_max_size:
                continue
            if len(self.last_episode_trajectory) <= self.targeted_edge_length * 2:
                # sample random pairs
                state_1_index = random.randint(0, self.targeted_edge_length * 2 - 1)
                state_2_index = random.randint(0, self.targeted_edge_length * 2 - 1)
                distance = abs(state_2_index - state_1_index)
                state_1 = self.last_episode_trajectory[state_1_index]
                state_2 = self.last_episode_trajectory[state_2_index]
            else:
                # Sample random pairs by sampling a distance around targeted edge length
                distance = np.random.poisson(self.targeted_edge_length)
                if state_index + distance >= len(self.last_episode_trajectory):
                    continue
                state_1 = observation
                state_2 = self.last_episode_trajectory[state_index + distance]
            """
            state_1_index = random.randint(0, len(self.last_episode_trajectory) - 1)
            state_2_index = random.randint(0, len(self.last_episode_trajectory) - 1)
            distance = abs(state_2_index - state_1_index)
            state_1 = self.last_episode_trajectory[state_1_index]
            state_2 = self.last_episode_trajectory[state_2_index]
            label = 0 if distance > self.targeted_edge_length else 1
            if label == 0:
                if len(self.tc_replay_buffer_negative) > self.tc_buffer_max_size // 2:
                    self.tc_replay_buffer_negative.pop(0)
                self.tc_replay_buffer_negative.append((state_1, state_2, label))
            elif label == 1:
                if len(self.tc_replay_buffer_positive) > self.tc_buffer_max_size // 2:
                    self.tc_replay_buffer_positive.pop(0)
                self.tc_replay_buffer_positive.append((state_1, state_2, label))

            """
            label = 0 if distance > self.targeted_edge_length else 1
            sample = (state_1, state_2, label)
            if len(self.tc_replay_buffer) >= self.tc_buffer_max_size:
                self.tc_replay_buffer[index] = sample
            else:
                self.tc_replay_buffer.append(sample)
            """

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
            inputs = torch.concat((states_1, states_2), dim=-1)

            """
            distances = torch.linalg.norm((states_1 - states_2), 2, dim=-1)
            for distance, label in zip(distances, labels):
                if label == 0:
                    if distance.item() in self.label_0_values:
                        self.label_0_values[distance.item()] += 1
                    else:
                        self.label_0_values[distance.item()] = 0
                elif label == 1:
                    if distance.item() in self.label_1_values:
                        self.label_1_values[distance.item()] += 1
                    else:
                        self.label_1_values[distance.item()] = 0
                else:
                    raise "Erreur"
            if self.episode_id != 0 and self.episode_id % 20 == 0:
                data = [[key, value] for key, value in self.label_0_values.items()]
                data.sort()
                plt.plot(np.array(data)[:, 0], np.array(data)[:, 1], color="blue", label="label 0")
                data = [[key, value] for key, value in self.label_1_values.items()]
                data.sort()
                plt.plot(np.array(data)[:, 0], np.array(data)[:, 1], color="green", label="label 1")
                plt.legend()
                plt.show()
            self.label_0_values = {}
            self.label_1_values = {}
            """

            # Predict label and learn loss
            predictions = self.topological_cognition_network(inputs).squeeze()
            labels = torch.Tensor(list(labels)).to(device=self.device, dtype=torch.float32)
            error = self.tc_criterion(predictions, labels)

            # self.topological_cognition_network.optimizer.zero_grad()
            # error.backward()
            # self.topological_cognition_network.optimizer.learning_step()

            self.topological_cognition_network.learn(error)

            """
            self.tc_errors_memory.append(error.detach().item())
            if len(self.tc_errors_memory) > 40:
                self.tc_average_errors_memory.append(mean(self.tc_errors_memory[-40:]))
            if self.episode_id != 0 and self.episode_id % 50 == 0:
                x = [i for i in range(len(self.tc_average_errors_memory))]
                plt.plot(x, self.tc_average_errors_memory, color="red", label="TC error")
                plt.legend()
                plt.show()
            """
