import os
import pickle

from utils import create_dir
from typing import Union

import networkx
from agents.global_planning.reachability_graph_learning.sorb import SORB
from agents.agent import Agent
import numpy as np
import networkx as nx
from gym.spaces import Box, Discrete


class SGM(SORB):

    """
    SGM agent as defined in sparse graphical memory paper (Emmons, 2020)
    https://proceedings.neurips.cc/paper/2020/file/385822e359afa26d52b5b286226f2cea-Paper.pdf.
    """

    name = "SGM"

    def __init__(self, state_space: Union[Box, Discrete], action_space: Union[Box, Discrete], **params):
        """
        @param state_space: Environment's state space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(state_space, action_space, **params)
        self.node_pruning_threshold = params.get("node_pruning_threshold", 2)
        self.max_edges_length = params.get("max_edges_length", 9)

        # Keep in memory the edge our agent is currently crossing, so we can prune it in case of failure.
        self.last_node_crossed = None
        self.target_node = None

    def build_graph(self):
        self.reachability_graph = networkx.DiGraph()
        states = self.control_policy.reinforcement_learning_agent.replay_buffer \
                    .sample(self.graph_building_buffer_size)[0][:, :-2].detach().cpu().numpy()

        # states = np.array([environment.reset()[0] for _ in range(self.nb_nodes_in_graph)])
        distances = self._get_pairwise_dist(states)

        # Select nodes to add as defined in
        added_states = []
        for state_id, state in enumerate(states):
            for node_state_id, node_state in added_states:
                dist_1 = distances[state_id * states.shape[0] + node_state_id]
                dist_2 = distances[node_state_id * states.shape[0] + state_id]
                if dist_1 < self.node_pruning_threshold or dist_2 < self.node_pruning_threshold:
                    break
            else:

                self.reachability_graph.add_node(state_id, state=state.copy()[self.state_to_goal_filter])

                # add edges from this new_state
                for node_state_id, node_state in added_states:
                    dist = distances[state_id * states.shape[0] + node_state_id]
                    if dist < self.max_edges_length:
                        self.reachability_graph.add_edge(state_id, node_state_id)

                    dist = distances[node_state_id * states.shape[0] + state_id]
                    if dist < self.max_edges_length:
                        self.reachability_graph.add_edge(node_state_id, state_id)

                # Add our new state to the list of added states
                added_states.append((state_id, state))

        self.graph_up_to_date = True

    def start_episode(self, state: np.ndarray, goal: np.ndarray, test_episode=False):
        Agent.start_episode(self, state, test_episode=test_episode)
        if self.default_state is None:
            self.default_state = state.copy()

        # Reset attributes that needs to
        self.current_goal = goal
        self.done = False
        self.nb_sub_task_interaction = 0
        self.last_node_crossed = None
        self.target_node = None

        if self.graph_up_to_date:
            current_node = self.get_node_for_state(state)
            final_node = self.get_node_for_state(self.current_goal, reachable_from=current_node)
            self.sub_goals = self.shortest_path(current_node, final_node)
            if self.sub_goals:
                self.next_goal = nx.get_node_attributes(self.reachability_graph, "state")[self.sub_goals[0]]
                self.target_node = self.sub_goals[0]
                assert isinstance(self.target_node, int)
            else:
                self.next_goal = self.current_goal
        else:
            self.next_goal = self.current_goal
        self.control_policy.start_episode(state, self.next_goal, test_episode=test_episode or self.graph_up_to_date)

    def process_interaction(self, action, reward, new_state, done, learn=True):
        Agent.process_interaction(self, action, reward, new_state, done, learn=not self.graph_up_to_date)
        pruned_edge = None
        if self.graph_up_to_date:
            if self.reached_sub_goal(new_state, self.next_goal):
                if self.target_node is not None:
                    self.last_node_crossed = self.target_node
                if self.sub_goals:
                    self.sub_goals.pop(0)
                if self.sub_goals:
                    self.next_goal = nx.get_node_attributes(self.reachability_graph, "state")[self.sub_goals[0]]
                    self.target_node = self.sub_goals[0]
                    assert isinstance(self.target_node, int)
                else:
                    self.next_goal = self.current_goal
                    self.target_node = None  # Keep in memory that we are not crossing an edge. Prevent pruning failures.
                self.nb_sub_task_interaction = 0
                self.control_policy.start_episode(new_state, self.next_goal,
                                                  test_episode=self.under_test or self.graph_up_to_date)
            else:
                self.nb_sub_task_interaction += 1
                if self.nb_sub_task_interaction >= self.max_interactions_per_sub_task:

                    """ This test and everything inside is the pruning procedure, specific to SGM. """
                    if learn and self.last_node_crossed is not None and self.target_node is not None:

                        # Prune the edge we tried to cross
                        self.reachability_graph.remove_edge(self.last_node_crossed, self.target_node)
                        pruned_edge = (self.last_node_crossed, self.target_node)
                        # Re-planing
                        self.start_episode(new_state, self.current_goal, test_episode=self.under_test)

                        # RE-PLANNING LOOP STOP CASE:
                        #   We can be sure that each episode finally stop because it exists at least one path in the
                        # graph our agent can cross entirely: the empty path.
                        # If every edge are not reachable, we will remove all of them and our agent will end up by
                        # trying to reach the final goal directly, which will lead to a re-planing in case of failure
                        # because self.last_node_crossed and self.target_node will be None.
                    else:
                        self.done = True
                assert (self.next_goal == self.control_policy.current_goal).all()
        else:
            self.control_policy.process_interaction(action, reward, new_state, done, learn=learn)
        assert (self.next_goal == self.control_policy.current_goal).all()
        return pruned_edge

    def save(self, directory):
        if directory[-1] != "/":
            directory += "/"
        create_dir(directory)

        # Save useful data to rebuild the class
        with open(directory + "state_space.pkl", 'wb') as f:
            pickle.dump(self.state_space, f)
        with open(directory + "action_space.pkl", 'wb') as f:
            pickle.dump(self.action_space, f)

        # Recursive save on our control policy
        self.control_policy.save(directory + "control_policy/")

        # Save graph
        with open(directory + "reachability_graph.pkl", "wb") as f:
            pickle.dump(self.reachability_graph, f)

    def load(self, directory):
        if directory[-1] != "/":
            directory += "/"
        assert os.path.isdir(directory)
        with open(directory + "state_space.pkl", 'rb') as f:
            self.state_space = pickle.load(f)
        with open(directory + "action_space.pkl", 'rb') as f:
            self.action_space = pickle.load(f)

        # Recursive save on our control policy
        self.control_policy.load(directory + "control_policy/")

        # Save graph
        with open(directory + "reachability_graph.pkl", "rb") as f:
            self.reachability_graph = pickle.load(f)
