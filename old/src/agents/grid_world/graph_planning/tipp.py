import copy
import math

import networkx as nx
import torch

from src.agents.agent import Agent
from src.agents.grid_world.graph_planning.topological_graph_planning_agent import PlanningTopologyLearner, \
    TopologyLearnerMode


class TIPP(PlanningTopologyLearner):
    def __init__(self, **params):
        params["name"] = "RGL"

        self.bowl_q_value_min = None
        self.bowl_q_value_max = None

        self.edges_similarity_threshold = params.get("edges_similarity_threshold", 0.3)
        self.nodes_similarity_threshold = params.get("nodes_similarity_threshold", 0.5)
        self.s_g_s = []
        self.s_g_f = []
        super().__init__(**params)

    def on_pre_training_done(self, start_state, reached_goals):
        self.bowl_q_value_min = None
        self.bowl_q_value_max = None
        start_state_ = copy.deepcopy(start_state)

        for goal in reached_goals:
            value = self.goal_reaching_agent.get_q_value(start_state_, goal)
            if self.bowl_q_value_min is None or self.bowl_q_value_min > value:
                self.bowl_q_value_min = value
            if self.bowl_q_value_max is None or self.bowl_q_value_max < value:
                self.bowl_q_value_max = value

    def get_similarity_approximation(self, state_1, goal):
        """
        Use the UVFA to get a value function approximation between two states.
        """
        q_value_approximation = self.goal_reaching_agent.get_q_value(state_1, goal)
        min_max_diff = self.bowl_q_value_max - self.bowl_q_value_min
        return (q_value_approximation - self.bowl_q_value_min) / min_max_diff

    def extend_graph(self):
        """
        Update the topology using the exploration trajectory.
        Precondition: An exploration trajectory has been made.
        """
        assert self.last_exploration_trajectory != []

        for state in self.last_exploration_trajectory:
            linkable_nodes = []
            for node_id, node_parameters in self.topology.nodes(data=True):
                node_position = node_parameters["state"]  # Position of node in the observation space
                similarity = self.get_similarity_approximation(state, node_position)
                # print("similarity = " + str(similarity))
                if similarity > self.nodes_similarity_threshold:
                    break
                if similarity > self.edges_similarity_threshold:
                    linkable_nodes.append(node_id)
            else:
                # => this observation is far enough from any nodes
                if linkable_nodes:  # Prevent to create unliked nodes
                    # Create node
                    new_node = self.create_node(state)

                    # Create edges
                    if self.last_node_explored not in linkable_nodes:
                        linkable_nodes.append(self.last_node_explored)
                    for node_id in linkable_nodes:
                        self.create_edge(new_node, node_id)

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        learn = self.mode != TopologyLearnerMode.GO_TO
        if self.random_exploration_steps_left is not None:
            assert self.mode == TopologyLearnerMode.LEARN_ENV
            self.last_exploration_trajectory.append(new_state)
            self.random_exploration_steps_left -= 1
            if self.random_exploration_steps_left == 0:
                if self.verbose:
                    print("Finished random exploration. We're done with this episode")
                self.extend_graph()
                self.done = True
            else:
                if self.verbose:
                    print("We continue random exploration for " + str(self.random_exploration_steps_left)
                          + " more time steps.")
        elif self.nb_trial_out_graph_left is not None:
            assert self.mode == TopologyLearnerMode.GO_TO
            self.nb_trial_out_graph_left -= 1
            if self.nb_trial_out_graph_left == 0:
                if self.verbose:
                    print("We're done trying to reach the final goal.")
                self.done = True
            elif self.verbose:
                print("We continue to reach global goal for " + str(self.nb_trial_out_graph_left)
                      + " more time steps.")
        else:  # We are trying to reach a sub-goal
            reached = self.reached(new_state)
            reward = 1 if reached else 0
            self.goal_reaching_agent.on_action_stop(action, new_state, reward, done=reached, learn=False)

            if reached:
                # The next sub-goal have been reached, we can remove it and continue to the next one
                if self.last_node_passed is not None and learn:
                    self.on_edge_crossed(self.last_node_passed, self.next_node_way_point)
                self.last_node_passed = self.next_node_way_point
                self.next_node_way_point = self.get_next_node_waypoint()
                self.current_subtask_steps = 0
                self.goal_reaching_agent.on_episode_stop()
                if self.next_node_way_point is None:
                    self.on_path_done(new_state)
                    if self.verbose:
                        print("Path is done.")
                else:
                    if self.verbose:
                        print("Reached a way point. Next one is " + str(self.next_node_way_point) + ".")
                    self.next_goal = self.get_goal_from_node(self.next_node_way_point)
                    self.goal_reaching_agent.on_episode_start(new_state, self.next_goal)
            else:
                self.current_subtask_steps += 1
                if self.current_subtask_steps > self.max_steps_to_reach:
                    # We failed reaching the next waypoint
                    if self.last_node_passed is not None and learn:
                        self.on_edge_failed(self.last_node_passed, self.next_node_way_point)
                        # self.topology.edges[self.last_node_passed, self.next_node_way_point]["cost"] = float('inf')
                    if self.verbose:
                        print("We failed reaching this way point ... We're done with this episode.")
                    self.done = True
                else:
                    if self.verbose:
                        print("Trying to reach way point " + str(self.next_node_way_point) + ". Time steps left = "
                              + str(self.max_steps_to_reach - self.current_subtask_steps))

        # Increment the counter of the node related to 'new_state'.
        self.topology.nodes[self.get_node_for_state(new_state)]["reached"] += 1
        Agent.on_action_stop(self, action, new_state, reward, done)

        if self.verbose:
            print("Interaction: observation=" + str(self.last_state) + ", action=" + str(action) + ", new_state="
                  + str(new_state) + ", agent goal=" + str(self.goal_reaching_agent.current_goal))

    def on_edge_crossed(self, last_node_passed, next_node_way_point):
        edge_attributes = self.topology.edges[last_node_passed, next_node_way_point]
        edge_attributes["cost"] = max(1, edge_attributes["cost"] / 2)

        states = nx.get_node_attributes(self.topology, 'state')
        s_g = states[last_node_passed] - states[next_node_way_point]
        self.s_g_s.append(s_g)

    def on_edge_failed(self, last_node_passed, next_node_way_point):
        edge_attributes = self.topology.edges[last_node_passed, next_node_way_point]

        states = nx.get_node_attributes(self.topology, 'state')
        s_g = states[last_node_passed] - states[next_node_way_point]
        self.s_g_f.append(s_g)

        try:
            edge_attributes["cost"] = math.exp(edge_attributes["cost"])
        except:
            pass

    def create_edge(self, first_node, second_node, potential=True, **params):
        """
        Create an edge between the two given nodes. If potential is True, it means that the edge weight will be lower in
        exploration path finding (to encourage the agent to test it) or higher in go to mode (to bring a lower
        probability of failure).
        """
        attributes = copy.deepcopy(self.edges_attributes)
        for key, value in params.items():
            attributes[key] = value
        if potential:
            attributes["cost"] = 1.
        self.topology.add_edge(first_node, second_node, **attributes)

    def shortest_path(self, node_from, node_to_reach):
        return nx.shortest_path(self.topology, node_from, node_to_reach, "cost")