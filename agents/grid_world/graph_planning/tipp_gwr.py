"""
GWR is a topological graph building strategy, that build a graph that fit the topology of a dataset.
https://www.sciencedirect.com/science/article/pii/S0893608002000783
"""
import math
import numpy as np
from old.src.agents.grid_world.graph_planning.topological_graph_planning_agent import PlanningTopologyLearner


class TIPP_GWR(PlanningTopologyLearner):
    def __init__(self, **params):
        self.Sw = params.get("Sw", 0.1)
        self.Sn = params.get("Sn", 0.05)
        self.age_max = params.get("age_max", 30)
        self.activity_threshold = params.get("activity_threshold", 0.88)
        # activity_threshold=0.89 for maps 0, 2, 5, 6
        # activity_threshold=0.93 for map 7
        # activity_threshold=0.94 for map 9
        # activity_threshold=0.96 for map 10
        self.firing_threshold = params.get("firing_threshold", 0.05)
        if "nodes_attributes" not in params.keys():
            params["nodes_attributes"] = {}
        params["nodes_attributes"]["win_count"] = 0  # Set the win_count default value in default nodes attributes.
        if "edges_attributes" not in params.keys():
            params["edges_attributes"] = {}
        params["edges_attributes"]["potential"] = True
        params["edges_attributes"]["exploration_cost"] = 0.
        params["edges_attributes"]["go_to_cost"] = float("inf")
        # '-> True if we think the two nodes are reachable, but unverified.
        params["name"] = "TIPP_GWR"
        super().__init__(**params)

    def extend_graph(self):
        """
        Update the topology using the exploration trajectory.
        Precondition: An exploration trajectory has been made.
        """
        assert self.last_exploration_trajectory != []

        for state in self.last_exploration_trajectory:
            if len(self.topology.nodes()) == 1:
                node_params = self.topology.nodes[0]
                node_distance = np.linalg.norm(node_params["state"] - self.get_position(state), 2)
                activity = math.exp(- node_distance)
                node_count_ratio = math.exp(- node_params["win_count"])
                node_params["state"] += self.Sw * (node_count_ratio * 4) * \
                                        (self.get_position(state) - node_params["state"])
                node_params["win_count"] += 1
                if activity < self.activity_threshold and node_count_ratio < self.firing_threshold:
                    # Add new node
                    new_node = self.create_node((node_params["state"] + self.get_position(state)) / 2)
                    self.create_edge(0, new_node, potential=False)
                continue
            first_node, first_node_params, first_node_distance, second_node, second_node_params, second_node_distance \
                = self.get_winners(state)
            first_node_params["win_count"] += 1

            # Link the two closest nodes or set their age to 0 if already linked
            self.create_edge(first_node, second_node, potential=True)

            # Compute the activity of the best matching unit
            activity = math.exp(- first_node_distance)
            first_node_count_ratio = math.exp(- first_node_params["win_count"])
            second_node_count_ratio = math.exp(- second_node_params["win_count"])
            if activity < self.activity_threshold and first_node_count_ratio < self.firing_threshold:
                # Add new node
                new_node_weights = (first_node_params["state"] + self.get_position(state)) / 2
                new_node = self.create_node(new_node_weights)
                self.create_edge(first_node, new_node, potential=not first_node == self.last_node_explored)
                self.create_edge(second_node, new_node, potential=not first_node == self.last_node_explored)
            else:
                # If a new node is not added, adapt the position of the winner and its neighbors
                first_node_params["state"] += \
                    self.Sw * (first_node_count_ratio * 4) * (self.get_position(state) - first_node_params["state"])
                second_node_params["state"] += \
                    self.Sn * (second_node_count_ratio * 4) * (self.get_position(state) - second_node_params["state"])

    def get_winners(self, state):
        """
        Return the two closest nodes from the given observation. The first node of the resulting tuple is the closest.
        If data is true, each node returned is a tuple with the node id and it's parameters.
        """
        first_node_id, second_node_id, first_node_distance, second_node_distance = None, None, None, None
        first_node_parameters, second_node_parameters = None, None

        for node_id, node_parameters in self.topology.nodes(data=True):
            node_state = node_parameters["state"]
            node_distance = self.distance(node_state, self.get_position(state))
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

    def distance(self, first_state, second_state):
        assert isinstance(first_state, np.ndarray)
        assert isinstance(second_state, np.ndarray)
        return np.linalg.norm(first_state - second_state, 2)

