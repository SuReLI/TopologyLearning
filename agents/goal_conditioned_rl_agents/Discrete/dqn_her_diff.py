from random import randrange

from agents.goal_conditioned_rl_agents.Discrete.gc_dqn import GCDQNAgent


class DQNHERAgent(GCDQNAgent):
    """
    A DQN agent that use HER.
    """
    def __init__(self, state_space, action_space, name="DQN + HER",
                 gamma=0.98, epsilon_min=0.01, epsilon_max=1., epsilon_decay_period=1000, epsilon_decay_delay=20,
                 buffer_size=1000000, learning_rate=0.0003, tau=0.001, batch_size=125,
                 layer_1_size=125, layer_2_size=100, nb_gradient_steps=1):

        super().__init__(state_space, action_space, name=name, gamma=gamma, epsilon_min=epsilon_min,
                         epsilon_max=epsilon_max, epsilon_decay_period=epsilon_decay_period,
                         epsilon_decay_delay=epsilon_decay_delay, buffer_size=buffer_size, learning_rate=learning_rate,
                         tau=tau, batch_size=batch_size, layer_1_size=layer_1_size,
                         layer_2_size=layer_2_size, nb_gradient_steps=nb_gradient_steps)

        # HER will relabel samples in the last trajectory. To do it, we need to keep this last trajectory in a memory
        self.last_trajectory = []
        # ... and store relabelling parameters
        self.nb_resample_per_states = 4

    def action(self, state):
        return super().action(state)

    def on_episode_start(self, *args):
        self.last_trajectory = []
        return super().on_episode_start(*args)

    def on_action_stop(self, action, new_state, reward, done):
        self.last_trajectory.append((self.last_state, action))
        return super().on_action_stop(action, new_state, reward, done)

    def on_episode_stop(self):
        # Relabel last trajectory
        if len(self.last_trajectory) <= self.nb_resample_per_states:
            return
        # For each state seen :
        for state_index, (state, action) in enumerate(self.last_trajectory[:-4]):
            new_state_index = state_index + 1
            new_state, _ = self.last_trajectory[new_state_index]

            # sample four goals in future states
            for relabelling_id in range(self.nb_resample_per_states):
                goal_index = randrange(new_state_index, len(self.last_trajectory))
                goal, _ = self.last_trajectory[goal_index]
                reward = (new_state_index / goal_index) * 2 - 1
                self.replay_buffer.append((state, action, reward, new_state, goal_index == new_state_index, goal))

    def reset(self):
        self.__init__(self.state_space, self.action_space, name=self.name, gamma=self.gamma,
                      epsilon_min=self.epsilon_min, epsilon_max=self.epsilon_max,
                      epsilon_decay_period=self.epsilon_decay_period, epsilon_decay_delay=self.epsilon_decay_delay,
                      buffer_size=self.buffer_size, learning_rate=self.learning_rate,
                      tau=self.tau, batch_size=self.batch_size,
                      layer_1_size=self.layer_1_size, layer_2_size=self.layer_2_size,
                      nb_gradient_steps=self.nb_gradient_steps)
