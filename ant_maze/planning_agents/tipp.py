import copy
import math
import os.path
import pickle

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from agents import Agent
from utils.sys_fun import get_red_green_color
from .topological_graph_planning_agent import PlanningTopologyLearner, TopologyLearnerMode


class TIPP(PlanningTopologyLearner):
    def __init__(self, **params):
        params["name"] = "RGL"

        self.bowl_q_value_min = None
        self.bowl_q_value_max = None

        self.edges_similarity_threshold = params.get("edges_similarity_threshold", 0.7)
        self.nodes_similarity_threshold = params.get("nodes_similarity_threshold", 0.5)
        self.s_g_s = []
        self.s_g_f = []

        super().__init__(**params)

        # Call directly because for ant_maze, pre-train result (weights and reached goals) are stored in pickle files.
        self.on_pre_training_done()

    def on_pre_training_done(self):
        """
        r = 11.7
        gran = r / 20
        points = []
        values = []
        initial_qpos = np.array([0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        state = np.concatenate((np.array([0., 0.]), initial_qpos, np.zeros(14)))
        for i in range(40):
            for j in range(40):
                a = (i - 19.5) * gran
                b = (j - 19.5) * gran
                points.append([a, b])
                goal = np.concatenate((np.array([a, b]), state[2:5]))

                # Compute angle to the goal.

                x = (i - 19.5) / 20
                y = (j - 19.5) / 20
                angle = math.acos(x) if y > 0 else -math.acos(x)
                state[6] = angle / math.pi
                action = self.goal_reaching_agent.action(state, goal)
                values.append(self.goal_reaching_agent.layers[0].critic.get_Q_value(state[np.newaxis], goal[np.newaxis], action[np.newaxis]))

        mini = min(values)
        maxi = max(values)
        colors = [get_red_green_color((v - mini) / (maxi - mini), hexadecimal=False) for v in values]
        points = np.array(points)
        colors = np.array(colors) / 255
        plt.scatter(points[:, 0], points[:, 1], c=colors)
        plt.show()
        """

        # Load start states and goals made during pre-train
        with open(os.path.dirname(__file__) + '/reached_sub_goals.pkl', 'rb') as f:
            archived_goals = pickle.load(f)

        self.bowl_q_value_min = None
        self.bowl_q_value_max = None
        initial_qpos = np.array([0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        state = np.concatenate((np.array([0., 0.]), initial_qpos, np.zeros(14)))
        for goal in archived_goals:
            value = self.goal_reaching_agent.get_q_value(state, goal)
            if self.bowl_q_value_min is None or self.bowl_q_value_min > value:
                self.bowl_q_value_min = value
            if self.bowl_q_value_max is None or self.bowl_q_value_max < value:
                self.bowl_q_value_max = value

    def get_similarity_approximation(self, state, goal):
        """
        Use the UVFA to get a value function approximation between two states.
        """
        q_value_approximation = self.goal_reaching_agent.get_q_value(state, goal)
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
                node_position = node_parameters["state"]  # Position of node in the state space
                similarity = self.get_similarity_approximation(state, node_position)
                # print("similarity = " + str(similarity))
                if similarity > self.nodes_similarity_threshold:
                    break
                if similarity > self.edges_similarity_threshold:
                    linkable_nodes.append(node_id)
            else:
                # => this state is far enough from any nodes
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

            if reached:
                self.nb_successes_on_edges += 1
                # The next sub-goal have been reached, we can remove it and continue to the next one
                if self.last_node_passed is not None and learn:
                    self.on_edge_crossed(self.last_node_passed, self.next_node_way_point)
                self.last_node_passed = self.next_node_way_point
                self.next_node_way_point = self.get_next_node_waypoint()
                self.current_subtask_steps = 0
                if self.next_node_way_point is None:
                    self.on_path_done(new_state)
                    if self.verbose:
                        print("Path is done.")
                else:
                    if self.verbose:
                        print("Reached a way point. Next one is " + str(self.next_node_way_point) + ".")
                    self.next_goal = self.get_goal_from_node(self.next_node_way_point)
            else:
                self.current_subtask_steps += 1
                if self.current_subtask_steps > self.max_steps_to_reach:
                    # We failed reaching the next waypoint
                    if self.last_node_passed is not None and learn:
                        self.on_edge_failed(self.last_node_passed, self.next_node_way_point)
                        # self.topology.edges[self.last_node_passed, self.next_node_way_point]["cost"] = float('inf')
                    self.nb_failures_on_edges += 1
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
            print("Interaction: state=" + str(self.last_state) + ", action=" + str(action) + ", new_state="
                  + str(new_state))

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
