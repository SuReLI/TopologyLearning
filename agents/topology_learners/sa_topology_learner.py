from enum import Enum

from agents.goal_conditioned_rl_agents.Discrete.dqn_her_diff import DqnHerDiffAgent
from agents.goal_conditioned_rl_agents.Discrete.dqn_her import DQNHERAgent
from agents.topology_learners.topology_learner import TopologyLearner
from agents.gc_agent import GoalConditionedAgent


class TopologyLearnerMode(Enum):
    LEARN_ENV = 1
    GO_TO = 2
    PATROL = 3


class SingleAgentTL(TopologyLearner):
    """
    An agent that is trained to learn the environment topology, so that learn by interacting with its environment, but
    don't need to reach a goal to do so. Then, he is able to exploit his knowledge of his environment to reach goals or
    to patrol inside it.

    This version use a single agent to reach every sub-goals.
    """

    def __init__(self, **params):
        state_space = params.get("state_space")
        action_space = params.get("action_space")
        device = params.get("device")
        goal_reaching_agent_class = params.get("goal_reaching_agent_class", DqnHerDiffAgent)
        self.goal_reaching_agent = goal_reaching_agent_class(state_space, action_space, device)
        TopologyLearner.__init__(self, **params)

    def on_simulation_start(self):
        super().on_simulation_start()
        self.goal_reaching_agent.on_simulation_start()

    def current_agent(self) -> GoalConditionedAgent:
        return self.goal_reaching_agent
