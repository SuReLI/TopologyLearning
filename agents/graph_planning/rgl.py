import copy
import math

import networkx as nx
import numpy as np
import torch

from ..agent import Agent
from .topological_graph_planning_agent import PlanningTopologyLearner, TopologyLearnerMode


class RGL(PlanningTopologyLearner):
    def __init__(self, **params):
        params["name"] = params.get("name", "RGL")

        self.distance_estimation_max = None

        self.edges_distance_threshold = params.get("edges_distance_threshold", 0.7)
        self.nodes_distance_threshold = params.get("nodes_distance_threshold", 0.5)
        super().__init__(**params)

    def on_pre_training_done(self, start_state, reached_goals):
        """
        Compute the longer distance estimation over every goal that has been reached during the pre-training.
        It allows to choose reachability parameters more easily.
        """
        self.distance_estimation_max = None
        start_state_ = copy.deepcopy(start_state)
        for goal in reached_goals:
            distance_estimation = - self.goal_reaching_agent.get_q_value(start_state_, goal)
            if self.distance_estimation_max is None or self.distance_estimation_max < distance_estimation:
                self.distance_estimation_max = distance_estimation

    def get_distance_estimation(self, state, goal):
        """
        Use the UVFA to get a value function approximation between two states.
        """

        if goal.shape[-1] == self.state_size:
            # When we compute a backward distance (from a node to a state) then goal can have a state shape and the
            # state chan have a goal shape, then can lead to a neural network input with a too small size in
            # the goal reaching policy's get_q_value function. we should make the state looks like a real state.
            state = np.concatenate((state, goal[self.goal_size:]))
            goal = goal[:self.goal_size]
        q_value_approximation = self.goal_reaching_agent.get_q_value(state, goal)
        distance_estimation = - q_value_approximation
        normalised_distance_estimation = distance_estimation / self.distance_estimation_max
        return normalised_distance_estimation

    def extend_graph(self):
        """
        Update the topology using the exploration trajectory.
        Precondition: An exploration trajectory has been made.
        """
        assert self.last_exploration_trajectory != []

        for state in self.last_exploration_trajectory:
            links_to_do = []
            for node_id, node_parameters in self.topology.nodes(data=True):
                node_position = node_parameters["state"]  # Position of node in the observation space
                forward_estimated_distance = self.get_distance_estimation(state, node_position)
                backward_estimated_distance = self.get_distance_estimation(node_position, state)
                if forward_estimated_distance < self.nodes_distance_threshold or \
                        backward_estimated_distance < self.nodes_distance_threshold:
                    break
                if forward_estimated_distance < self.edges_distance_threshold:
                    links_to_do.append({"node": node_id, "forward": True, "cost": forward_estimated_distance})
                if backward_estimated_distance < self.edges_distance_threshold:
                    links_to_do.append({"node": node_id, "forward": False, "cost": backward_estimated_distance})
            else:
                # => this observation is far enough from any nodes
                if links_to_do:  # Prevent to create unliked nodes
                    # Create node
                    new_node = self.create_node(state)

                    # Create edges
                    if self.last_node_explored not in [link_to_do["node"] for link_to_do in links_to_do]:
                        last_node_state = self.topology.nodes[self.last_node_explored]["state"]
                        backward_estimated_distance = self.get_distance_estimation(last_node_state, state)
                        links_to_do.append({"node": self.last_node_explored,
                                            "forward": False, "cost": backward_estimated_distance})
                    for link_to_do in links_to_do:
                        if link_to_do["forward"]:
                            self.create_edge(new_node, link_to_do["node"], cost=link_to_do["cost"])
                        else:
                            self.create_edge(link_to_do["node"], new_node, cost=link_to_do["cost"])

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

    def get_states_distance(self, state_1, state_2):
        return self.get_distance_estimation(state_1, state_2)

    def on_edge_crossed(self, last_node_passed, next_node_way_point):
        edge_attributes = self.topology.edges[last_node_passed, next_node_way_point]
        edge_attributes["cost"] = max(1, edge_attributes["cost"] / 2)

    def on_edge_failed(self, last_node_passed, next_node_way_point):
        edge_attributes = self.topology.edges[last_node_passed, next_node_way_point]
        edge_attributes["cost"] = float("inf")

    def create_edge(self, first_node, second_node, **params):
        """
        Create an edge between the two given nodes. If potential is True, it means that the edge weight will be lower in
        exploration path finding (to encourage the agent to test it) or higher in go to mode (to bring a lower
        probability of failure).
        """
        attributes = copy.deepcopy(self.edges_attributes)
        for key, value in params.items():
            attributes[key] = value
        attributes["cost"] = params.get("cost", 1.)  # Set to one only if it's unset.
        self.topology.add_edge(first_node, second_node, **attributes)

    def shortest_path(self, node_from, node_to_reach):
        return nx.shortest_path(self.topology, node_from, node_to_reach, "cost")