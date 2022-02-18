from gym.spaces import Box
from agents.gc_agent import GoalConditionedAgent


class GCAgentContinuous(GoalConditionedAgent):

    def __init__(self, state_space, action_space, device, name="Random policy"):
        super().__init__(state_space, action_space, device, name)
        assert isinstance(action_space, Box)
        if len(action_space.shape) > 1:
            raise NotImplementedError
        self.lower_bound = action_space.low[0]
        self.higher_bound = action_space.high[0]
        self.actions_range = self.higher_bound - self.lower_bound

    def reset(self):
        self.__init__(self.state_space, self.action_space, self.device, self.name)
