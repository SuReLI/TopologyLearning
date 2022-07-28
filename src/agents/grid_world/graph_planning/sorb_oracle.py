import numpy as np
import torch

from src.agents import DQNHERAgent, DqnHerDiffAgent
import networkx as nx
from random import sample
from src.settings import settings


class SORB(DqnHerDiffAgent):
    def __init__(self, **params):
        self.oracle = params.get("oracle")
        self.nb_nodes = params.get("nb_nodes", 200)
        self.nb_trials_on_edge = params.get("nb_trials_on_edge", 30)
        self.nb_trials_on_edge_left = self.nb_trials_on_edge
        self.nb_trial_out_graph = params.get("random_exploration_duration", 70)
        self.nb_trial_out_graph_left = None
        self.topology = nx.Graph()
        self.final_goal = None
        self.done = False

        self.bowl_q_value_min = None
        self.bowl_q_value_max = None
        self.reached_goals = None
        self.start_state = None
        self.sub_goals = []

        self.tolerance_margin = params.get("tolerance_margin", (0., 0.))
        if isinstance(self.tolerance_margin, tuple):
            self.tolerance_margin = np.array(self.tolerance_margin)
        default_tolerance_radius = np.mean(self.tolerance_margin) * 1.1
        self.tolerance_radius = params.get("tolerance_radius", default_tolerance_radius)

        self.edges_similarity_threshold = 0.70
        self.nb_episodes_before_edges_update = 20
        self.nb_episodes_left_before_edges_update = self.nb_episodes_before_edges_update
        super().__init__(**params)

    def on_episode_start(self, *args):
        self.done = False
        self.nb_trial_out_graph_left = None
        self.nb_trials_on_edge_left = self.nb_trials_on_edge
        state, self.final_goal = args
        assert self.final_goal is not None
        if self.topology.nodes:
            last_sub_goal = self.get_node_for_state(self.final_goal, reachable_only=True)
            sub_goals = nx.shortest_path(self.topology, 0, last_sub_goal)
            self.sub_goals = [self.topology.nodes[node]["state"] for node in sub_goals]
            super().on_episode_start(state, self.sub_goals[0])
        else:
            self.nb_trial_out_graph_left = settings.episode_length
            super().on_episode_start(state, self.final_goal)
        if self.reached_goals is not None:
            self.nb_episodes_left_before_edges_update -= 1
            if self.nb_episodes_left_before_edges_update == 0:
                self.update_edges()
                self.nb_episodes_left_before_edges_update = self.nb_episodes_before_edges_update

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        if self.nb_trial_out_graph_left:
            self.nb_trial_out_graph_left -= 1
            if self.nb_trial_out_graph_left == 0:
                self.done = True
            if (new_state == self.final_goal).all():
                reward = 1
                self.done = True
            else:
                reward = -1
            done = self.done
        else:
            reward = -1
            if self.reached(new_state):
                self.sub_goals.pop(0)
                self.nb_trials_on_edge_left = self.nb_trials_on_edge
                reward = 1
                if not self.sub_goals:
                    self.nb_trial_out_graph_left = self.nb_trial_out_graph
                    self.current_goal = self.final_goal
                else:
                    self.current_goal = self.sub_goals[0]
            else:
                self.nb_trials_on_edge_left -= 1
                if self.nb_trials_on_edge_left <= 0:
                    self.done = True
        super().on_action_stop(action, new_state, reward, done or self.done, learn=learn)

    def reached(self, state: np.ndarray, goal: np.ndarray = None) -> bool:
        if goal is None:
            goal = self.sub_goals[0]
        dist = np.linalg.norm(goal - state, 2)
        return dist < self.tolerance_radius

    def on_pre_training_done(self, start_state, reached_goals):
        self.reached_goals = reached_goals
        self.start_state = start_state
        self.init_graph()

    def get_node_for_state(self, state, data=False, reachable_only=False):
        """
        Select the node that best represent the given observation.
        """
        assert isinstance(state, np.ndarray)
        if not self.topology.nodes:
            return None  # The graph  hasn't been initialised yet.
        node_data = None
        closest_distance = None
        for node_id, args in self.topology.nodes(data=True):
            if reachable_only:
                try:
                    nx.shortest_path(self.topology, 0, node_id)
                except:
                    continue
            distance = np.linalg.norm(args["state"] - state, 2)
            if closest_distance is None or distance < closest_distance:
                node_data = (node_id, args)
                closest_distance = distance
        return node_data if data else node_data[0]

    def get_similarity_approximation(self, state, goal):
        """
        Use the UVFA to get a value function approximation between two states.
        """
        q_value_approximation = torch.max(self.model(state - goal)).detach().item()
        min_max_diff = self.bowl_q_value_max - self.bowl_q_value_min
        return (q_value_approximation - self.bowl_q_value_min) / min_max_diff

    def init_graph(self):
        """
        Iterate through every pair of nodes in the graph, compute their similarity, link them is their similarity is
        above self.edges_similarity_threshold.
        """
        # Build nodes
        nodes_states = sample(self.oracle, self.nb_nodes)
        for id, state in enumerate(nodes_states):
            self.topology.add_node(id, state=state)

        # Build edges
        self.update_edges()

    def update_edges(self):
        # Update q_value bounds
        self.bowl_q_value_min = None
        self.bowl_q_value_max = None
        for state in self.reached_goals:
            value = torch.max(self.model(self.start_state - state)).detach().item()
            if self.bowl_q_value_min is None or self.bowl_q_value_min > value:
                self.bowl_q_value_min = value
            if self.bowl_q_value_max is None or self.bowl_q_value_max < value:
                self.bowl_q_value_max = value

        # Update edges
        for first_node_id, first_node_parameters in self.topology.nodes(data=True):
            for second_node_id, second_node_parameters in self.topology.nodes(data=True):
                if first_node_id == second_node_id:
                    continue
                first_state = first_node_parameters["state"]
                second_state = second_node_parameters["state"]
                similarity = self.get_similarity_approximation(first_state, second_state)
                if similarity > self.edges_similarity_threshold:
                    self.topology.add_edge(first_node_id, second_node_id)

    def get_position(self, state):
        if state.shape[0] == 2:
            return state
        else:
            return state[:2]


