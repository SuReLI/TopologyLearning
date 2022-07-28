from random import randrange
import numpy as np
from src.agents.ant_maze.graph_free.goal_conditioned_sac import GoalConditionedSACAgent
from src.settings import settings


class SACHERAgent(GoalConditionedSACAgent):
    """
    Soft Actor Critic (SAC) + Hindsight Experience Replay (HER => Goal conditioned agent)
    """

    def __init__(self, **params):
        params["name"] = params.get("name", "SAC + HER")  # Set if not already set
        super().__init__(**params)

        # HER will relabel samples in the last trajectory. To do it, we need to keep this last trajectory in a memory
        self.last_trajectory = []
        # ... and store relabelling parameters
        self.nb_resample_per_states = 4

        self.state_to_goal_filter = params.get("state_to_goal_filter", [1, 1, 0, 0])
        self.state_to_goal_filter = np.array(self.state_to_goal_filter).astype(np.bool)

    def on_episode_start(self, *args):
        self.last_trajectory = []
        return super().on_episode_start(*args)

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        if learn:
            self.last_trajectory.append((self.last_state, action))
        super().on_action_stop(action, new_state, reward, done, learn=learn)

    def on_episode_stop(self):
        # Relabel last trajectory
        if len(self.last_trajectory) <= self.nb_resample_per_states:
            return
        # For each observation seen :

        for state_index, (state, action) in enumerate(self.last_trajectory[:-4]):
            new_state_index = state_index + 1
            new_state, _ = self.last_trajectory[new_state_index]

            # sample four goals in future states
            for relabelling_id in range(self.nb_resample_per_states):
                goal_index = randrange(new_state_index, len(self.last_trajectory))
                goal, _ = self.last_trajectory[goal_index]
                reached = goal_index == new_state_index

                reach_bonus = 1.0
                common_malus = 0.0
                reward = reach_bonus if reached else common_malus

                self.replay_buffer.append((state, action, reward, new_state, reached, goal[self.state_to_goal_filter]))

        super().on_episode_stop()
