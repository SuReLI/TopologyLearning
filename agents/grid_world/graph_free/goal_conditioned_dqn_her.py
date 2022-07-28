from random import randrange

from old.src.agents.grid_world.graph_free.goal_conditioned_dqn import GCDQNAgent


class DQNHERAgent(GCDQNAgent):
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation.
    """
    """
    An agent that learn an approximated Q-Function using a neural network.
    This Q-Function is used to find the best action to execute in a given observation, in order to reach
    a goal, given at the beginning of the episode.
    """

    def __init__(self, **params):
        params["name"] = params.get("name", "DQN + HER")  # Set if not already set
        super().__init__(**params)

        # HER will relabel samples in the last trajectory. To do it, we need to keep this last trajectory in a memory
        self.last_trajectory = []
        # ... and store relabelling parameters
        self.nb_resample_per_states = 4

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
                goal = goal[:2]
                reward = (new_state_index / goal_index) * 2 - 1
                self.replay_buffer.append((state, action, reward, new_state, goal_index == new_state_index, goal))
        super().on_episode_stop()
