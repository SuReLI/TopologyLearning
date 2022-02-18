from agents.topology_learners.sc_topology_learner import SkillChainingTL
from agents.topology_learners.sa_topology_learner import SingleAgentTL, TopologyLearnerMode
from agents.topology_learners.topology_learner import TopologyLearner


class DensityFirstTL(TopologyLearner):
    """
    A topology learner that use a node density first strategy for node selection for graph exploration.
    It means that it will select the node with the lower density, and then try to reach it.
    The density of a node is the number of time the agent goes into its influence area.
    """

    def __init__(self, **params):
        if "nodes_attributes" in params:
            params["nodes_attributes"]["density"] = 0
        else:
            params["nodes_attributes"] = {"density": 0}
        self.current_exploration_nodes_path = []
        super().__init__(**params)

    def on_episode_start(self, *args):
        state, mode, _ = args[:3]
        if mode.value == TopologyLearnerMode.LEARN_ENV.value:
            self.set_exploration_path(state)
        super().on_episode_start(*args)

    def get_exploration_next_node(self):
        if self.current_exploration_nodes_path:
            result = self.current_exploration_nodes_path.pop(0)
            return result
        # Otherwise, it will return None, aka. no nodes left.

    def set_exploration_path(self, state):
        """
        Select or generate a new goal that is promising for exploration.
        """
        if not self.topology.nodes:
            self.current_exploration_nodes_path = []
            return
        node_from = self.get_node_for_state(state)

        # Get the node with the lowest density
        best_node = min(self.topology.nodes(data=True), key=lambda x: x[1]["density"])[0]

        # Find the shortest path through our topology to it
        shortest_path = self.shortest_path(node_from, best_node)

        # Stor the chosen path
        self.current_exploration_nodes_path = shortest_path


class DFSkillChainingTL(SkillChainingTL, DensityFirstTL):
    """
    A topology learner that use a node density first strategy for node selection for graph exploration.
    It means that it will select the node with the lower density, and then try to reach it.
    The density of a node is the number of time the agent goes into its influence area.
    """

    def __init__(self, **params):
        SkillChainingTL.__init__(self, **params)
        DensityFirstTL.__init__(self, **params)


class DFSingleAgentTL(SingleAgentTL, DensityFirstTL):
    """
    A topology learner that use a node density first strategy for node selection for graph exploration.
    It means that it will select the node with the lower density, and then try to reach it.
    The density of a node is the number of time the agent goes into its influence area.
    """

    def __init__(self, **params):
        SingleAgentTL.__init__(self, **params)
        DensityFirstTL.__init__(self, **params)
