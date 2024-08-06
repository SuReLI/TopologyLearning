# Goal conditioned agent
from random import randrange
import numpy as np
from .goal_conditioned_value_based_agent import GoalConditionedValueBasedAgent


class HER(GoalConditionedValueBasedAgent):
    """
    A global agent class for goal conditioned agents. The # NEW tag indicate differences between Agent class and this
    one.
    """

    def __init__(self, reinforcement_learning_agent_class, state_space, action_space, **params):
        super().__init__(reinforcement_learning_agent_class, state_space, action_space, **params)
        self.last_trajectory = []
        # ... and store relabelling parameters
        self.nb_resample_per_states = 4
        self.name = self.reinforcement_learning_agent.name + " + HER"

    def start_episode(self, state: np.ndarray, goal: np.ndarray, test_episode=False):
        self.last_trajectory = []
        return super().start_episode(state, goal, test_episode)

    def process_interaction(self, action, new_state, reward, done, learn=True):
        if learn and not self.under_test:
            self.last_trajectory.append((self.last_state, action))
        super().process_interaction(action, new_state, reward, done, learn=learn)

    def stop_episode(self):
        # Relabel last trajectory
        if self.under_test or len(self.last_trajectory) <= self.nb_resample_per_states:
            return

        # For each observation seen :
        for state_index, (state, action) in enumerate(self.last_trajectory[:-self.nb_resample_per_states]):
            new_state_index = state_index + 1
            new_state, _ = self.last_trajectory[new_state_index]

            # sample four goals in future states
            for relabelling_id in range(self.nb_resample_per_states):
                goal_index = randrange(new_state_index, len(self.last_trajectory))
                target_state, _ = self.last_trajectory[goal_index]
                goal = self.state_to_goal(target_state)

                features = self.get_features(state, goal)
                # Compute a reward that goes from -1, for the first state of the fake trajectory, to 0, if the
                # new_state if the fake goal.
                reward = (new_state_index / goal_index) - 1
                new_features = self.get_features(new_state, goal)

                done = goal_index == new_state_index
                self.reinforcement_learning_agent.save_interaction(features, action, reward, new_features, done)
        super().stop_episode()
