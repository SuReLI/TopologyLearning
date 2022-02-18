from enum import Enum

from agents.graph_building_strategies.gwr import GWR
from agents.topology_learners.topology_learner import TopologyLearner
from agents.goal_conditioned_rl_agents.Discrete.sac_discrete.sac_her import SACHERAgent


class TopologyLearnerMode(Enum):
    LEARN_ENV = 1
    GO_TO = 2
    PATROL = 3


def instantiate_agent(**params):
    """
    Used to instantiate an agent when a node is created.
    """
    agent = SACHERAgent(**params)
    agent.on_simulation_start()
    return agent


class SkillChainingTL(TopologyLearner):
    """
    An agent that is trained to learn the environment topology, so that learn by interacting with its environment, but
    don't need to reach a goal to do so. Then, he is able to exploit his knowledge of his environment to reach goals or
    to patrol inside it.

    This version use multiple agent to reach every sub-goals. We here use une agent per node in the graph, that manage
    to reach neighbors nodes.
    """

    def __init__(self, **params):
        action_space = params.get("action_space")
        state_space = params.get("state_space")
        device = params.get("device")
        goal_reaching_agent_class = params.get("goal_reaching_agent_class", SACHERAgent)

        agent_attribute = (instantiate_agent,
                           {  # Call parameters
                               "state_space": state_space,
                               "action_space": action_space,
                               "device": device,
                               "layer_1_size": 64,
                               "layer_2_size": 32,
                               "batch_size": 64,
                               "max_buffer_size": 10000
                           })
        if "nodes_attributes" in params:
            params["nodes_attributes"]["agent"] = agent_attribute
        else:
            params["nodes_attributes"] = {"agent": agent_attribute}

        super().__init__(**params)

        self.goal_reaching_agent_class = goal_reaching_agent_class
        self.start_agent = goal_reaching_agent_class(state_space, action_space, device)
        self.next_node_id = 0

    def current_agent(self) -> SACHERAgent:
        if self.last_node_passed is None:
            return self.start_agent
        else:
            return self.node_agent(self.last_node_passed)

    def node_agent(self, node_id) -> SACHERAgent:
        return self.topology.nodes[node_id]["agent"]

    # Transfer call to the embedded goal reaching agent
    def on_simulation_stop(self):
        super().on_simulation_stop()
        for _, node_params in self.topology.nodes(data=True):
            node_params["agent"].on_simulation_stop()
