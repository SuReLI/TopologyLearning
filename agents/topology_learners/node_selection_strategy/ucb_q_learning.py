from math import sqrt, log

from agents.topology_learners.topology_learner import TopologyLearner
from agents.topology_learners.sa_topology_learner import SingleAgentTL
from agents.topology_learners.sc_topology_learner import SkillChainingTL


class UCB_QL(TopologyLearner):
    """
    A topology learner that use a node explorations first strategy for node selection for graph exploration.
    It means that it will select the node with the lower explorations, and then try to reach it.
    The explorations of a node is the number of time the agent goes into its influence area.
    """

    def __init__(self, **params):

        self.ucb_exploration_ratio = params.get("ucb_exploration_ratio", 0.7)
        self.gamma = params.get("gamma", 0.98)
        self.learning_rate = params.get("learning_rate", 0.1)

        if "nodes_attributes" in params:
            params["nodes_attributes"]["explorations"] = 0
            params["nodes_attributes"]["q_values"] = {}
            params["nodes_attributes"]["nb_selections"] = {}
        else:
            params["nodes_attributes"] = {"explorations": 0, "q_values": {}, "nb_selections": {}}

        super().__init__(**params)

    def get_nb_choices(self, node):
        nb_selections: dict = self.topology.nodes[node]["nb_selections"]
        result = 0
        if nb_selections:
            for value in nb_selections.values():
                result += value
        return result

    def get_nb_selections(self, current_node, next_node):
        """
        Verify if the nb_selections value is set for this edge (current_node, next_node), and set it to 0 otherwise.
        Then the value is returned.
        """
        nb_selections = self.topology.nodes[current_node]["nb_selections"]
        if next_node not in nb_selections.keys():
            nb_selections[next_node] = 0
        return nb_selections[next_node]

    def get_ucb_exploration_term(self, node, next_node):
        nb_choices = self.get_nb_choices(node)
        nb_selections = self.get_nb_selections(node, next_node)
        if nb_selections == 0:
            return float('inf')
        lg = log(nb_choices + 1)
        sq = sqrt(lg / nb_selections)
        res = self.ucb_exploration_ratio * sq
        return res

    def is_local_minimum(self, node):
        explorations = self.topology.nodes[node]["explorations"]
        for neighbor in self.topology.neighbors(node):
            neighbor_explorations = self.topology.nodes[neighbor]["explorations"]
            if explorations > neighbor_explorations:
                return False
        return True

    def get_q_value(self, current_node, next_node):
        """
        Verify if the Q-value is set for this edge (current_node, next_node), and set it to 0 otherwise.
        Then the value is returned.
        """
        q_values = self.topology.nodes[current_node]["q_values"]
        if next_node not in q_values.keys():
            q_values[next_node] = 0
        return q_values[next_node]

    def update_q_value(self, node, next_node, reward):
        next_nodes_q_values = self.topology.nodes[next_node]["q_values"]
        if not next_nodes_q_values:
            best_next_q_value = 0
        else:
            best_next_q_value = max(next_nodes_q_values.values())

        target_q_value = reward + self.gamma * best_next_q_value
        new_q_val = (1 - self.learning_rate) * self.get_q_value(node, next_node) \
            + self.learning_rate * target_q_value
        self.topology.nodes[node]["q_values"][next_node] = new_q_val

    def edge_choice(self, node_from, next_node):
        if next_node not in self.topology.nodes[node_from]["nb_selections"]:
            self.topology.nodes[node_from]["nb_selections"][next_node] = 1
        else:
            self.topology.nodes[node_from]["nb_selections"][next_node] += 1

    def get_exploration_next_node(self):
        # If we are in a graph local minimum, we can stop and start a random exploration.
        if self.last_node_passed is None:
            self.topology.nodes[0]["explorations"] += 1
            return 0

        if self.is_local_minimum(self.last_node_passed):
            if self.verbose:
                print("££££££££££££££££££££")
                print("  UCB choice report:")
                print(" We are in a local minimum.")
                print(" current node = " + str(self.last_node_passed) + " with explorations = "
                      + str(self.topology.nodes[self.last_node_passed]["explorations"]))
                for neighbor in self.topology.neighbors(self.last_node_passed):
                    print(" neighbor " + str(neighbor) + ": explorations = " +
                          str(self.topology.nodes[neighbor]["explorations"]))
                print("££££££££££££££££££££")
            return None

        # Find the best next node according to Q-values and UCB exploration term
        best_utility = None
        best_utility_node = None
        utility_memory = {}
        q_values_memory = {}
        exploration_memory = {}
        for neighbor in self.topology.neighbors(self.last_node_passed):
            q_value = self.get_q_value(self.last_node_passed, neighbor)
            exploration = self.get_ucb_exploration_term(self.last_node_passed, neighbor)
            neighbor_utility = q_value + exploration
            utility_memory[neighbor] = neighbor_utility
            q_values_memory[neighbor] = q_value
            exploration_memory[neighbor] = exploration
            if best_utility is None or best_utility < neighbor_utility:
                best_utility = neighbor_utility
                best_utility_node = neighbor

        if self.verbose:
            print("££££££££££££££££££££")
            print("  UCB choice report:")
            print(" We choose next node " + str(best_utility_node))
            for neighbor in self.topology.neighbors(self.last_node_passed):
                print(" - Neighbor " + str(neighbor) + ": "
                      + "utility=" + str(utility_memory[neighbor])
                      + "; q_value=" + str(q_values_memory[neighbor])
                      + "; exploration=" + str(exploration_memory[neighbor])
                      + "; nb_selected=" + str(self.get_nb_selections(self.last_node_passed, neighbor))
                      + "; nb_selected_total=" + str(self.get_nb_choices(self.last_node_passed)))
            print("££££££££££££££££££££")

        # In crease the counter of how many times we took this edge from the current node
        if best_utility_node not in self.topology.nodes[self.last_node_passed]["nb_selections"]:
            self.topology.nodes[self.last_node_passed]["nb_selections"][best_utility_node] = 1
        else:
            self.topology.nodes[self.last_node_passed]["nb_selections"][best_utility_node] += 1

        self.topology.nodes[best_utility_node]["explorations"] += 1
        return best_utility_node

    def on_reaching_waypoint_failed(self, last_node, next_node):
        super().on_reaching_waypoint_failed(last_node, next_node)
        # Update the Q-value using a reward of 0 (because we failed)
        if last_node is not None:
            self.update_q_value(last_node, next_node, reward=0)

    def on_reaching_waypoint_succeed(self, last_node, next_node):
        super().on_reaching_waypoint_succeed(last_node, next_node)
        if last_node is not None:
            # Update the Q-value using a reward of 1 / explorations (because we reached the next node)
            # This reward is supposed to be more important when we are close to the border of our topology.
            explorations = self.topology.nodes[next_node]["explorations"]
            reward = 1 / explorations
            self.update_q_value(last_node, next_node, reward=reward)


class UCBSkillChainingTL(SkillChainingTL, UCB_QL):
    """
    A topology learner that use a node explorations first strategy for node selection for graph exploration.
    It means that it will select the node with the lower explorations, and then try to reach it.
    The explorations of a node is the number of time the agent goes into its influence area.
    """

    def __init__(self, **params):
        SkillChainingTL.__init__(self, **params)
        UCB_QL.__init__(self, **params)


class UCBSingleAgentTL(SingleAgentTL, UCB_QL):
    """
    A topology learner that use a node explorations first strategy for node selection for graph exploration.
    It means that it will select the node with the lower explorations, and then try to reach it.
    The explorations of a node is the number of time the agent goes into its influence area.
    """

    def __init__(self, **params):
        SingleAgentTL.__init__(self, **params)
        UCB_QL.__init__(self, **params)
