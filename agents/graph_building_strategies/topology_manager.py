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

    def create_node(self, weights, **params):
        attributes = copy.deepcopy(self.nodes_attributes)
        for key, value in params.items():
            attributes[key] = value
        for key, value in attributes.items():
            if isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
                # Here, the value of this parameter should be initialised using a function call.
                # The value inside self.nodes_attributes is a tuple, with the function in first attribute, and it's
                # parameters as a dict in the second.
                function = value[0]
                parameters_dict = value[1]
                attributes[key] = function(**parameters_dict)

        attributes["weights"] = weights
        self.topology.add_node(self.next_node_id, **attributes)
        self.next_node_id += 1
        return self.next_node_id - 1

    def create_edge(self, first_node, second_node, **params):
        attributes = copy.deepcopy(self.edges_attributes)
        for key, value in params.items():
            attributes[key] = value
        self.topology.add_edge(first_node, second_node, **attributes)

    def get_node_for_state(self, state, data=False):
        """
        Select the node that best represent the given state.
        """
        assert isinstance(state, np.ndarray)
        if not self.topology.nodes:
            return None  # The graph  hasn't been initialised yet.
        closest = None
        node_data = None
        closest_distance = None
        for node_id, args in self.topology.nodes(data=True):
            distance = np.linalg.norm(args["weights"] - state, 2)
            if closest is None or distance < closest_distance:
                closest = node_id
                node_data = (node_id, args)
                closest_distance = distance
        return node_data if data else node_data[0]

    def remove_subgraph(self, node):
        to_remove = [n for n in self.topology.neighbors(node)]
        self.topology.remove_node(node)
        for node in to_remove:
            self.remove_subgraph(node)

    def remove_edge(self, node_1, node_2):
        try:
            self.topology.remove_edge(node_1, node_2)
        except:
            return
        try:
            nx.shortest_path(self.topology, 0, node_1)
        except:
            self.remove_subgraph(node_1)
        try:
            nx.shortest_path(self.topology, 0, node_2)
        except:
            self.remove_subgraph(node_2)

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


