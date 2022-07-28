from random import choice
from src.agents.grid_world.graph_free.goal_conditioned_dqn_her import DQNHERAgent
from src.settings import settings


class AutonomousDQNHERAgent(DQNHERAgent):
    """
    A DQN agent that use HER, and that is also able to choose its own goals, by sampling them from passes interactions.
    """
    def __init__(self, **params):
        self.done = False
        self.goals_buffer = []
        super().__init__(**params)

    def on_episode_start(self, *args):
        self.done = False
        state, goal = args
        if goal is None:  # Else it's a test episode
            goal = choice(self.goals_buffer) if self.goals_buffer else None
        super().on_episode_start(state, goal)

    def action(self, state):
        if self.current_goal is None:
            return self.action_space.sample()  # Randomly explore the environment
        else:
            return super().action(state)

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        self.goals_buffer.append(new_state)
        reached = (new_state == self.current_goal).all()
        reward = 1 if reached else -1
        if reached or self.episode_time_step_id > settings.episode_length:
            self.done = True
        learn = learn and self.current_goal is not None
        super().on_action_stop(action, new_state, reward, self.done, learn=learn)
