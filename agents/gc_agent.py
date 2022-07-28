# Goal conditioned agent

from old.src.agents.agent import Agent


class GoalConditionedAgent(Agent):
    """
    An global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.goal_size = params.get("goal_size", None)
        if self.goal_size is None:
            self.goal_size = self.state_size
        self.current_goal = None

    def on_episode_start(self, *args):
        state, goal = args
        super().on_episode_start(state)
        self.current_goal = goal
