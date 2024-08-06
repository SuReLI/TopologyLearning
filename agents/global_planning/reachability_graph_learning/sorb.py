from typing import Union

import networkx as nx
import numpy as np
from networkx import NetworkXNoPath

from agents.agent import Agent
from gym.spaces import Box, Discrete

from agents.discrete.distributional_dqn import DistributionalDQN
from agents.goal_conditioned_wrappers.goal_conditioned_value_based_agent import GoalConditionedValueBasedAgent
from agents.goal_conditioned_wrappers.her import HER
from agents.continuous.distributional_ddpg import DistributionalDDPG


class SORB(Agent):

    """
    SORB agent as defined in sorb paper (Eisenbach, 2017)
    https://proceedings.neurips.cc/paper/2019/file/5c48ff18e0a47baaf81d8b8ea51eec92-Paper.pdf.
    """

    name = "SORB"

    def __init__(self, state_space: Union[Box, Discrete], action_space: Union[Box, Discrete], **params):
        """
        @param state_space: Environment's state space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(state_space, action_space, **params)
        self.state_to_goal_filter = params.get("state_to_goal_filter",
                                               np.array([True, True] + [False] * (self.state_size - 2)))

        self.goal_space = Box(low=self.state_space.low[self.state_to_goal_filter],
                              high=self.state_space.high[self.state_to_goal_filter])
        assert isinstance(self.goal_space, Box) or isinstance(self.goal_space, Discrete)
        self.goal_size = self.goal_space.shape[0]
        self.goal_shape = self.goal_space.shape

        assert len(self.goal_shape) == 1, "Multi dimensional spaces are not supported."
        if isinstance(self.action_space, Box):
            self.control_policy: GoalConditionedValueBasedAgent = HER(DistributionalDDPG, state_space, action_space,
                                                                      goal_space=self.goal_space, **params)
        if isinstance(self.action_space, Discrete):
            self.control_policy: GoalConditionedValueBasedAgent = HER(DistributionalDQN, state_space, action_space,
                                                                      goal_space=self.goal_space,**params)
        assert hasattr(self.control_policy, "replay_buffer")
        self.current_goal = None

        # There is two phases in SORB training. First, the control policy is trained by trying to reach goals in the
        # environment. Then, we sample some states in the control policy's buffer, and build links between those that
        # are considered as reachable (check their paper for more details on that).
        # If the control policy learn again, the distance estimation (aka, the Q-value in SORB) change and the graph is
        # no more usable, because links should be computed again with the new distance function.

        self.graph_up_to_date = False
        #  '--> True if we didn't learn since the graph has been build. Set to false at each learning step.
        self.reachability_graph = nx.DiGraph()
        self.graph_building_buffer_size = params.get("nb_nodes", 1000)
        self.max_edges_length = params.get("max_edges_length", 5)
        self.reachability_threshold = params.get("reachability_threshold", 1)
        self.sub_goals = []
        self.next_goal = None
        self.done = False
        self.nb_sub_task_interaction = 0
        self.max_interactions_per_sub_task = 20
        self.default_state = None

    def reached_sub_goal(self, state, sub_goal):
        assert not isinstance(sub_goal, int)
        return np.linalg.norm(state[self.state_to_goal_filter] - sub_goal) < self.reachability_threshold

    def on_pre_training_done(self):
        self.build_graph()

    def build_graph(self):
        states = self.control_policy.reinforcement_learning_agent.replay_buffer \
                    .sample(self.graph_building_buffer_size)[0][:, 2:].detach().cpu().numpy()

        for node_id, state in enumerate(states):
            self.reachability_graph.add_node(node_id, state=state[self.state_to_goal_filter])

        distances = self._get_pairwise_dist(states)

        for node_1_id, state_1 in enumerate(states[:-1]):
            for node_2_id, state_2 in enumerate(states[node_1_id + 1:]):
                node_2_id += node_1_id + 1
                dist = distances[node_1_id * states.shape[0] + node_2_id]
                if dist < self.max_edges_length:
                    self.reachability_graph.add_edge(node_1_id, node_2_id)

                dist = distances[node_2_id * states.shape[0] + node_1_id]
                if dist < self.max_edges_length:
                    self.reachability_graph.add_edge(node_2_id, node_1_id)
        self.graph_up_to_date = True

    def _get_pairwise_dist(self, states):
        """
        Estimates the pairwise distances.
        """
        assert states.shape[0] > 1, "cannot compute pair wise distance on a single state."
        nb_states = states.shape[0]
        a = np.repeat(states, nb_states, axis=0)
        b = np.tile(states[:, self.state_to_goal_filter], (nb_states, 1))
        pair_wise_data = np.concatenate((a, b), -1)

        batch_max_length = int(2e6)
        nb_calls = pair_wise_data.shape[0] // batch_max_length \
                   + (0 if pair_wise_data.shape[0] % batch_max_length == 0 else 1)
        q_values = None
        for i in range(nb_calls):
            #   '--> A too big batch can lead to memory issues that crash the program.
            data = pair_wise_data[i * batch_max_length: (i + 1) * batch_max_length]
            q = self.control_policy.reinforcement_learning_agent\
                .get_value(data)
            q_values = np.concatenate((q_values, q), 0) if q_values is not None else q
        return - q_values

    def get_distance_estimation(self, state_1, state_2):
        features = np.concatenate((state_1, state_2), -1)
        return - self.control_policy.reinforcement_learning_agent.get_value(features)

    def shortest_path(self, node_from, node_to_reach):
        return nx.shortest_path(self.reachability_graph, node_from, node_to_reach)

    def get_node_for_state(self, state, data=False, reachable_from=None):
        """
        Select the node that best represent the given state.
        """
        assert isinstance(state, np.ndarray)
        if state.shape[0] == self.goal_size and self.goal_size < self.state_size:
            state = np.concatenate((state, self.default_state[state.shape[0]:]))
        if not self.reachability_graph.nodes:
            return None  # The graph  hasn't been initialised yet.
        node_data = None
        closest_distance = None
        for node_id, args in self.reachability_graph.nodes(data=True):
            if reachable_from is not None:
                try:
                    self.shortest_path(reachable_from, node_id)
                except NetworkXNoPath:
                    continue
            distance = self.get_distance_estimation(state, args["state"])
            if closest_distance is None or distance < closest_distance:
                node_data = (node_id, args)
                closest_distance = distance
        return node_data if data else node_data[0]

    def start_episode(self, state: np.ndarray, goal: np.ndarray, test_episode=False):
        super().start_episode(state, test_episode=test_episode)
        if self.default_state is None:
            self.default_state = state.copy()
        self.current_goal = goal
        self.done = False
        self.nb_sub_task_interaction = 0
        if self.graph_up_to_date:
            current_node = self.get_node_for_state(state)
            final_node = self.get_node_for_state(self.current_goal, reachable_from=current_node)
            self.sub_goals = self.shortest_path(current_node, final_node)
            if self.sub_goals:
                self.next_goal = nx.get_node_attributes(self.reachability_graph, "state")[self.sub_goals[0]]
            else:
                self.next_goal = self.current_goal
        else:
            self.next_goal = self.current_goal
        self.control_policy.start_episode(state, self.next_goal, test_episode=self.under_test or self.graph_up_to_date)

    def action(self, state, explore=True):
        return self.control_policy.action(state, explore=not self.graph_up_to_date)

    def process_interaction(self, action, reward, new_state, done, learn=True):
        super().process_interaction(action, reward, new_state, done, learn=learn)
        learn = learn and not self.graph_up_to_date

        if self.graph_up_to_date:
            if self.reached_sub_goal(new_state, self.next_goal):
                if self.sub_goals:
                    self.sub_goals.pop(0)
                if self.sub_goals:
                    self.next_goal = nx.get_node_attributes(self.reachability_graph, "state")[self.sub_goals[0]]
                else:
                    self.next_goal = self.current_goal
                self.nb_sub_task_interaction = 0
                self.control_policy.start_episode(new_state, self.next_goal,
                                                  test_episode=self.under_test or self.graph_up_to_date)
            else:
                self.nb_sub_task_interaction += 1
                if self.nb_sub_task_interaction >= self.max_interactions_per_sub_task:
                    self.done = True
                assert (self.next_goal == self.control_policy.current_goal).all()
        else:
            self.control_policy.process_interaction(action, reward, new_state, done, learn=learn)
        assert (self.next_goal == self.control_policy.current_goal).all()

    def learn_interaction(self, state, action, reward, new_state, done):
        pass
