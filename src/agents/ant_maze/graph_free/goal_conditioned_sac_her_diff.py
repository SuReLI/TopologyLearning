"""
A DQN agent that learn using Hindsight Experience Replay (HER) but that take s - g (with s the observation given by the 7
environment and g the goal our agent is trying to reach) as the agent's observation.
"""

import copy
from copy import deepcopy
import numpy as np
import torch
from torch import optim
from torch.nn import ReLU

from src.agents.ant_maze.graph_free.goal_conditioned_sac_her import SACHERAgent
from src.agents.utils.mlp import MLP
from torch.nn import functional


class SACHERDiffAgent(SACHERAgent):
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
        params["name"] = params.get("name", "Diff SAC + HER")  # Set if not already set
        super().__init__(**params)

        self.actor = MLP(self.state_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(), 2 * self.nb_actions,
                         learning_rate=self.actor_lr, optimizer_class=optim.Adam, device=self.device).float()
        self.target_actor = copy.deepcopy(self.actor)

        self.critic_1 = MLP(self.state_size + self.nb_actions, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                            1, learning_rate=self.critic_lr, optimizer_class=optim.Adam,
                            device=self.device).float()
        self.target_critic_1 = copy.deepcopy(self.critic_1)

        self.critic_2 = MLP(self.state_size + self.nb_actions, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                            1, learning_rate=self.critic_lr, optimizer_class=optim.Adam,
                            device=self.device).float()
        self.target_critic_2 = copy.deepcopy(self.critic_2)

    def state_goal_difference(self, state, goal, output_as_numpy=False):
        """
        Convert state into an s - g state :
        Ex.: s = [x_s, y_s, a, b, c, ...]; g = [x_g, y_g]
          s' = [x_s - x_g, y_s - y_g, a, b, c, ...]

        This function work on batches.
        """
        assert len(state.shape) == len(goal.shape)
        assert goal.shape[-1] == len(np.where(self.state_to_goal_filter)[0])
        assert state.shape[-1] == len(self.state_to_goal_filter)

        index_to_shrink = np.where(~self.state_to_goal_filter)[0]
        index_to_shrink += np.arange(0, - index_to_shrink.shape[0], -1)
        if len(state.shape) == 1:
            filled_goal = np.insert(deepcopy(goal), index_to_shrink, np.nan)
        else:
            filled_goal = np.insert(deepcopy(goal), index_to_shrink, np.nan, axis=1)

        result = np.where(self.state_to_goal_filter, state - filled_goal, state)
        return result if output_as_numpy else torch.from_numpy(result).to(self.device)

    def get_q_value(self, state, goal):
        with torch.no_grad():
            goal_conditioned_state = self.state_goal_difference(state, goal)

            next_actions, _ = self.sample_action(goal_conditioned_state, actor_network=self.target_actor)
            critic_input = torch.cat((goal_conditioned_state, next_actions), -1)
            q_values_1 = self.target_critic_1.forward(critic_input).view(-1)
            q_values_2 = self.target_critic_2.forward(critic_input).view(-1)
            q_values = torch.min(q_values_1, q_values_2)
        return q_values

    def action(self, state):
        with torch.no_grad():
            state = self.state_goal_difference(state, self.current_goal)
            actions, _ = self.sample_action(state)
        return actions.cpu().detach().numpy()

    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            states, actions, rewards, new_states, done, goals = self.replay_buffer.sample(self.batch_size)

            states = self.state_goal_difference(states, goals)
            new_states = self.state_goal_difference(new_states, goals)

            # Training critic
            with torch.no_grad():
                next_actions, next_log_probs = self.sample_action(new_states, actor_network=self.target_actor)
                next_q_values_1 = self.target_critic_1.forward(np.concatenate((new_states, next_actions), -1)).view(-1)
                next_q_values_2 = self.target_critic_2.forward(np.concatenate((new_states, next_actions), -1)).view(-1)
                next_q_values = torch.min(next_q_values_1, next_q_values_2)

            q_hat = self.reward_scale * rewards + self.gamma * (1 - done) * next_q_values

            q_values_1 = self.critic_1.forward(torch.cat((states, actions), 1)).view(-1)
            critic_loss_1 = functional.mse_loss(q_values_1, q_hat)
            self.critic_1.learn(critic_loss_1)
            self.target_critic_1.converge_to(self.critic_1, tau=self.tau)

            q_values_2 = self.critic_2.forward(torch.cat((states, actions), 1)).view(-1)
            critic_loss_2 = functional.mse_loss(q_values_2, q_hat)
            self.critic_2.learn(critic_loss_2)
            self.target_critic_2.converge_to(self.critic_2, tau=self.tau)

            if self.learning_step % self.policy_update_frequency == 0:
                for _ in range(self.policy_update_frequency):
                    # Train actor
                    actions, log_probs = self.sample_action(states)
                    log_probs = log_probs.view(-1)
                    critic_values_1 = self.critic_1.forward(torch.cat((states, actions), -1)).view(-1)
                    critic_values_2 = self.critic_2.forward(torch.cat((states, actions), -1)).view(-1)
                    critic_values = torch.min(critic_values_1, critic_values_2)

                    actor_loss = torch.mean(self.actor_alpha * log_probs - critic_values)
                    self.actor.learn(actor_loss)
                    self.target_actor.converge_to(self.actor, tau=self.tau)
            self.learning_step += 1
