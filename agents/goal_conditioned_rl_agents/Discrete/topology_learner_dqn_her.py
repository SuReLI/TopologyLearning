from random import randrange, randint, choice
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

from agents.goal_conditioned_rl_agents.Discrete.dqn_her import DQNHERAgent
from agents.goal_conditioned_rl_agents.Discrete.gc_dqn import GCDQNAgent


class AutonomousDQNHERAgent(DQNHERAgent):
    """
    A DQN + HER agent that is able to choose its own goals using his passed explorations.
    """
    def __init__(self, state_space, action_space, name="DQN + HER", tolerance_radius=1e-6,
                 gamma=0.98, epsilon_min=0.01, epsilon_max=1., epsilon_decay_period=1000, epsilon_decay_delay=20,
                 buffer_size=1000000, learning_rate=0.0003, tau=0.001, batch_size=125,
                 layer_1_size=125, layer_2_size=100, nb_gradient_steps=1):

        super().__init__(state_space, action_space, name=name, gamma=gamma, epsilon_min=epsilon_min,
                         epsilon_max=epsilon_max, epsilon_decay_period=epsilon_decay_period,
                         epsilon_decay_delay=epsilon_decay_delay, buffer_size=buffer_size, learning_rate=learning_rate,
                         tau=tau, batch_size=batch_size, layer_1_size=layer_1_size,
                         layer_2_size=layer_2_size, nb_gradient_steps=nb_gradient_steps)
        self.tolerance_radius = tolerance_radius
        self.goals_buffer = []
        self.max_goals_buffer_size = 1000
        self.nb_states_seen = 0  # For reservoir sampling
        self.done = False
        self.max_iterations_per_episode = 100
        self.last_episodes_results = []
        self.last_episodes_results_mean_memory = []
        self.max_episodes_memory_length = 10
        self.sub_plots_shape = (2, 1)

    def on_episode_start(self, *args):
        self.done = False
        if len(args) == 1:
            state = args[0]
            if len(self.goals_buffer) == 0:
                goal = None
            else:
                goal = choice(self.goals_buffer)
        elif len(args) > 1:
            state, goal = args[:2]
        self.store_goal(state)
        return super().on_episode_start(state, goal)

    def action(self, state):
        if self.current_goal is None:
            # Mean that the buffer was empty and that's our first episode (we start by exploring)
            return self.action_space.sample()
        else:
            return super().action(state)

    def store_goal(self, goal):
        # Append the new state to our goals buffer using reservoir sampling
        if self.nb_states_seen < self.max_goals_buffer_size:
            self.goals_buffer.append(goal)
        else:
            index = randint(0, self.nb_states_seen)
            if index < self.max_goals_buffer_size:
                self.goals_buffer[index] = goal
        self.nb_states_seen += 1

    def reached(self, state: np.ndarray, goal: np.ndarray = None) -> bool:
        if goal is None:
            if self.current_goal is None:
                return False
            goal = self.current_goal
        distance = np.linalg.norm(goal - state, 2)
        return distance < self.tolerance_radius

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        self.store_goal(new_state)

        if self.current_goal is None:
            # We are doing a random exploration, no need to compute reward and learn directly on these interactions
            # until relabelling

            # Should be done by super().on_episode_stop() call, but we don't call it because we don't want to learn.
            self.episode_time_step_id += 1
            self.simulation_time_step_id += 1
            self.last_state = new_state

            if learn:
                self.last_trajectory.append((self.last_state, action))
            if self.episode_time_step_id >= self.max_iterations_per_episode:
                self.done = True
        else:
            reached = self.reached(new_state)
            if reached:
                if len(self.last_episodes_results) == self.max_episodes_memory_length:
                    self.last_episodes_results.pop(0)
                self.last_episodes_results.append(1)
                reward = 1.
                self.done = True
            else:
                reward = -1
                if self.episode_time_step_id >= self.max_iterations_per_episode:
                    if len(self.last_episodes_results) == self.max_episodes_memory_length:
                        self.last_episodes_results.pop(0)
                    self.last_episodes_results.append(0)
                    self.done = True
            return super().on_action_stop(action, new_state, reward, self.done, learn=learn)

    def on_episode_stop(self):
        if self.last_episodes_results:
            self.last_episodes_results_mean_memory.append(mean(self.last_episodes_results))

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

    def update_plots(self, environment, sub_plots):
        ax1, ax2 = sub_plots
        ax1.clear()
        ax2.clear()

        ax1.set_title("Agent POV goal reaching accuracy")
        ax1.plot(self.last_episodes_results_mean_memory)
