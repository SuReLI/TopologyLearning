from gym.spaces import Discrete
from agents.gc_agent import GoalConditionedAgent


class GCAgentDiscrete(GoalConditionedAgent):

    def __init__(self, state_space, action_space, device, name="discrete_actions_agent"):
        super().__init__(state_space, action_space, device, name)
        assert isinstance(action_space, Discrete)
        self.lower_bound = 0
        self.higher_bound = action_space.n
        self.actions_range = action_space.n

    def reset(self):
        self.__init__(self.state_space, self.action_space, self.device, self.name)
