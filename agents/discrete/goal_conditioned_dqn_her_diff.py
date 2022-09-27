"""
A DQN agent that learn using Hindsight Experience Replay (HER) but that take s - g (with s the observation given by the 7
environment and g the goal our agent is trying to reach) as the agent's observation.
"""

import copy
import numpy as np
import torch
from torch import optim
from torch.nn import ReLU

from agents.discrete.goal_conditioned_dqn_her import DQNHERAgent
from agents.utils.mlp import MLP


class DqnHerDiffAgent(DQNHERAgent):
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
        params["name"] = params.get("name", "Diff DQN + HER")  # Set if not already set
        super().__init__(**params)
        self.model = MLP(self.state_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(), self.nb_actions,
                         learning_rate=self.learning_rate, optimizer_class=optim.Adam, device=self.device).float()
        self.target_model = copy.deepcopy(self.model).to(self.device)

    def get_q_value(self, state, goal):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device) if isinstance(state, np.ndarray) else state
            goal = torch.from_numpy(goal).to(self.device) if isinstance(goal, np.ndarray) else goal

            if len(state.shape) == 1:
                goal_conditioned_state = torch.cat((state[:self.goal_size] - goal, state[self.goal_size:]), dim=-1)
                return torch.max(self.model(goal_conditioned_state)).detach().item()
            elif len(state.shape) == 2:
                goal_conditioned_state = torch.cat((state[:, :self.goal_size] - goal, state[:, self.goal_size:]), dim=-1)
                return torch.max(self.model(goal_conditioned_state), dim=-1).values
            else:
                raise NotImplementedError("Batch shape not supported, not implemented error.")

    def action(self, state):
        goal_conditioned_state = np.concatenate((state[:self.goal_size] - self.current_goal, state[self.goal_size:]))

        if self.simulation_time_step_id > self.epsilon_decay_delay:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        if np.random.rand() < self.epsilon:  # Epsilon greedy
            action = np.random.randint(self.nb_actions)
        else:
            # greedy_action(self.model, observation) function in RL5 notebook
            with torch.no_grad():
                q_values = self.model(goal_conditioned_state)
                action = torch.argmax(q_values).item()
        return action

    def learn(self):
        for _ in range(self.nb_gradient_steps):
            # gradient_step() function in RL5 notebook
            if len(self.replay_buffer) > self.batch_size:
                #  NEW, samples from buffer contains goals
                states, actions, rewards, new_states, dones, goals = self.replay_buffer.sample(self.batch_size)

                # NEW concatenate states and goals, because we need to put them inside our model
                goal_conditioned_states = np.concatenate((states[:, :self.goal_size] - goals,
                                                          states[:, self.goal_size:]), axis=-1)
                goal_conditioned_new_states = np.concatenate((new_states[:, :self.goal_size] - goals,
                                                              new_states[:, self.goal_size:]), axis=-1)

                q_prime = self.target_model(goal_conditioned_new_states).max(1)[0].detach()
                update = rewards + self.gamma * (1 - dones) * q_prime
                q_s_a = self.model(goal_conditioned_states).gather(1, actions.to(torch.long).unsqueeze(1))
                loss = self.criterion(q_s_a, update.unsqueeze(1))
                self.model.learn(loss)

        self.target_model.converge_to(self.model, self.tau)