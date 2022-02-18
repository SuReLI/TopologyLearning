"""
GWR (Growing When Require) is a famous state of the art neuronal gaz (NG) for topology learning. This NG is able to
grow (add new nodes) when new data are far from its closest node, and when this closest node cannot learn anymore.

This variant (GWReR) with eR for (with) exponential risk, increase the risk of a link using
"""
import math
import networkx as nx
import numpy as np
from agents.graph_building_strategies import GWR


class GWReR(GWR):
    def __init__(self, topology, distance_function=None, nodes_attributes=None, edges_attributes=None, Sw=0.1, Sn=0.05,
                 age_max=30, activity_threshold=0.85, firing_threshold=0.05):

        if edges_attributes is None:
            edges_attributes = {}
        edges_attributes["consecutive_failed"] = 0
        super().__init__(topology=topology, distance_function=distance_function, nodes_attributes=nodes_attributes,
                         edges_attributes=edges_attributes, Sw=Sw, Sn=Sn, age_max=age_max,
                         activity_threshold=activity_threshold, firing_threshold=firing_threshold)

    def shortest_path(self, node_from, node_to_reach):
        return nx.shortest_path(self.topology, node_from, node_to_reach, "risk")

    def on_reaching_waypoint_failed(self, last_node, next_node):
        if last_node is None:
            return
        edge_data = self.topology.get_edge_data(last_node, next_node)
        if edge_data is None:
            return
        data = self.topology.get_edge_data(last_node, next_node)
        data["consecutive_failed"] += 1
        data["risk"] = math.exp(data["consecutive_failed"])
        self.topology.add_edge(last_node, next_node, **data)

    def on_reaching_waypoint_succeed(self, last_node, next_node):
        if last_node is None:
            return
        edge_data = self.topology.get_edge_data(last_node, next_node)
        if edge_data is None:
            return
        data = self.topology.get_edge_data(last_node, next_node)
        data["consecutive_failed"] = 0
        data["risk"] = 1  # exp(0) = 1
        self.topology.add_edge(last_node, next_node, **data)
