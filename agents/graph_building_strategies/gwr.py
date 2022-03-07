"""
GWR (Growing When Require) is a famous state of the art neuronal gaz (NG) for topology learning. This NG is able to
grow (add new nodes) when new data are far from its closest node, and when this closest node cannot learn anymore.
"""
import math
from statistics import mean

import networkx as nx
import numpy as np

from agents.graph_building_strategies.topology_manager import TopologyManager


class GWR(TopologyManager):
    def __init__(self, topology, distance_function=None, nodes_attributes=None, edges_attributes=None, Sw=0.1, Sn=0.05,
                 age_max=30, activity_threshold=0.88, firing_threshold=0.05):
        # activity_threshold=0.87 for map 7
        # activity_threshold=0.94 for map 9
        # activity_threshold=0.96 for map 10
        if edges_attributes is None:
            edges_attributes = {}
        edges_attributes["risk"] = 1.
        edges_attributes["density"] = 0
        if nodes_attributes is None:
            nodes_attributes = {}
        nodes_attributes["win_count"] = 0
        super().__init__(topology, distance_function, nodes_attributes, edges_attributes)
        self.Sw = Sw
        self.Sn = Sn
        self.age_max = age_max
        self.activity_threshold = activity_threshold
        self.firing_threshold = firing_threshold
        self.failed_edges = []

    def shortest_path(self, node_from, node_to_reach):
        return nx.shortest_path(self.topology, node_from, node_to_reach, "risk")

    def on_new_data(self, data):
        """
        Update the topology using the given state
        """
        # data_ = []
        # for elt in data:
        #     data_ += elt[1]
        # data = data_
        data = data[-1][1]
        """
        data = data[-1][1]
        """

        for state in data:
            if len(self.topology.nodes()) == 1:
                node_params = self.topology.nodes[0]
                node_distance = np.linalg.norm(node_params["weights"] - state, 2)
                activity = math.exp(- node_distance)
                node_count_ratio = math.exp(- node_params["win_count"])
                node_params["weights"] += self.Sw * (node_count_ratio * 4) * (state - node_params["weights"])
                node_params["win_count"] += 1
                if activity < self.activity_threshold and node_count_ratio < self.firing_threshold:
                    # Add new node
                    new_node = self.create_node((node_params["weights"] + state) / 2)
                    self.create_edge(0, new_node)
                continue
            first_node, first_node_params, first_node_distance, second_node, second_node_params, second_node_distance \
                = self.get_winners(state)
            first_node_params["win_count"] += 1

            # Link the two closest nodes or set their age to 0 if already linked
            self.reset_age(first_node, second_node)

            # Compute the activity of the best matching unit
            activity = math.exp(- first_node_distance)
            first_node_count_ratio = math.exp(- first_node_params["win_count"])
            second_node_count_ratio = math.exp(- second_node_params["win_count"])
            if activity < self.activity_threshold and first_node_count_ratio < self.firing_threshold:
                # Add new node
                new_node_weights = (first_node_params["weights"] + state) / 2
                new_node = self.create_node(new_node_weights)
                self.create_edge(first_node, new_node)
                self.create_edge(second_node, new_node)
            else:
                # If a new node is not added, adapt the position of the winner and its neighbors
                first_node_params["weights"] += \
                    self.Sw * (first_node_count_ratio * 4) * (state - first_node_params["weights"])
                second_node_params["weights"] += \
                    self.Sn * (second_node_count_ratio * 4) * (state - second_node_params["weights"])

            # Increment age of all links emanating from the winner
            """
            to_remove = []
            for neighbor_id in self.topology.neighbors(first_node):
                edge_data = self.topology.get_edge_data(neighbor_id, first_node)
                if edge_data["age"] >= self.age_max:
                    if (neighbor_id, first_node) not in to_remove:
                        to_remove.append((neighbor_id, first_node))
                else:
                    edge_data["age"] += 1
            for neighbor_id, first_node in to_remove:
                self.topology.remove_edge(neighbor_id, first_node)
                self.remove_if_isolated(neighbor_id)
                self.remove_if_isolated(first_node)
            """

    def reset_age(self, first_node, second_node):
        edge_data = self.topology.get_edge_data(first_node, second_node)
        if edge_data is None:
            # We should create the link, with every parameter initialised
            self.create_edge(first_node, second_node)
        else:
            # If we re-create an already existing edge, all his parameters will be reset. But we just want to reset the
            # age.
            self.topology.add_edge(first_node, second_node, age=0)

    def get_winners(self, state):
        """
        Return the two closest nodes from the given state. The first node of the resulting tuple is the closest.
        If data is true, each node returned is a tuple with the node id and it's parameters.
        """
        first_node_id, second_node_id, first_node_distance, second_node_distance = None, None, None, None
        first_node_parameters, second_node_parameters = None, None

        for node_id, node_parameters in self.topology.nodes(data=True):
            node_weights = node_parameters["weights"]
            node_distance = self.distance_function(node_weights, state)
            if first_node_distance is None or first_node_distance > node_distance:
                second_node_id = first_node_id
                second_node_distance = first_node_distance
                second_node_parameters = first_node_parameters
                first_node_id = node_id
                first_node_distance = node_distance
                first_node_parameters = node_parameters
            elif second_node_distance is None or second_node_distance > node_distance:
                second_node_id = node_id
                second_node_distance = node_distance
                second_node_parameters = node_parameters
        return first_node_id, first_node_parameters, first_node_distance, \
            second_node_id, second_node_parameters, second_node_distance

    def on_reaching_waypoint_failed(self, last_node, next_node):
        if last_node is None:
            return
        edge_data = self.topology.get_edge_data(last_node, next_node)
        if edge_data is None:
            return
        data = self.topology.get_edge_data(last_node, next_node)
        data["risk"] = float('inf')
        self.topology.add_edge(last_node, next_node, **data)
        if (last_node, next_node) in self.failed_edges:
            print("Failed on edge already failed before")
        else:
            self.failed_edges.append((last_node, next_node))
            self.failed_edges.append((next_node, last_node))

    def on_reaching_waypoint_succeed(self, last_node, next_node):
        """if last_node is None:
            return
        edge_data = self.topology.get_edge_data(last_node, next_node)
        if edge_data is None:
            return
        edge_data["risk"] = 1
        self.topology.add_edge(last_node, next_node, **edge_data)"""
