"""
The topology manager is used by a topology learner to build the topology depending on the data we give to it.
"""
import copy

import networkx as nx
import numpy as np


def euclidian_distance(state1, state2):
    return np.linalg.norm(state1 - state2, 2)


class TopologyManager:
    def __init__(self, topology, distance_function=None, nodes_attributes=None, edges_attributes=None):
        """
         - Topology: the topology we need to manage
         - nodes_attributes: attributes and default value of a newly born node.
            [(attr1, def_value), (attr2, def_value), ...]
         - edges_attributes: same than nodes_attributes but for edges
        """
        assert isinstance(topology, nx.Graph)
        if nodes_attributes is None:
            nodes_attributes = dict()
        if edges_attributes is None:
            edges_attributes = dict()
        self.nodes_attributes = nodes_attributes
        self.edges_attributes = edges_attributes
        self.topology = topology
        self.next_node_id = 0
        self.distance_function = distance_function if distance_function is not None else euclidian_distance

    def shortest_path(self, node_from, node_to_reach):
        return nx.shortest_path(self.topology, node_from, node_to_reach)

    def create_node(self, weights):
        params = copy.deepcopy(self.nodes_attributes)
        for key, value in self.nodes_attributes.items():
            if isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
                # Here, the value of this parameter should be initialised using a function call.
                # The value inside self.nodes_attributes is a tuple, with the function in first attribute, and it's
                # parameters as a dict in the second.
                function = value[0]
                parameters_dict = value[1]
                params[key] = function(**parameters_dict)

        params["weights"] = weights
        self.topology.add_node(self.next_node_id, **params)
        self.next_node_id += 1
        return self.next_node_id - 1

    def create_edge(self, first_node, second_node):
        self.topology.add_edge(first_node, second_node, **self.edges_attributes)

    def on_new_data(self, data):
        """
        Learn the topology using the given data.
        data shape:
         [ (None, list of states),
           (node_id, list of states),
           (node_id, list of states),
           ... ]
        list of tuple, the first element of each tuple is the last node we passed before to generate the states inside
        the second element of each tuples, a list of states.
        The first element of the first tuple is None because at the beginning, we didn't reach any nodes. We do action
        and reach states in order to reach the first node.
        """
        pass

    def on_reaching_waypoint_failed(self, last_node, next_node):
        pass

    def on_reaching_waypoint_succeed(self, last_node, next_node):
        pass


