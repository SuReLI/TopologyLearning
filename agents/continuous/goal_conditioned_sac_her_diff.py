# Goal conditioned deep Q-network

import copy
import pickle
from random import randrange
from torch.nn import ReLU
from ..utils.mlp import MLP
from ..utils.replay_buffer import ReplayBuffer
from torch import optim
from torch.nn import functional
from torch.distributions.normal import Normal

import torch
import numpy as np

from ..gc_agent import GoalConditionedAgent
from agents.continuous.goal_conditioned_sac_her import GoalConditionedSacHerAgent


class GoalConditionedSacHerDiffAgent(GoalConditionedSacHerAgent):

    def __init__(self, **params):
        params["name"] = params.get("name", "SAC Diff + HER")
        GoalConditionedAgent.__init__(self, **params)

        self.actor_lr = params.get("actor_lr", 0.001)
        self.critic_lr = params.get("critic_lr", 0.001)
        alpha = params.get("alpha", None)
        self.critic_alpha = params.get("critic_alpha", 0.05)
        self.actor_alpha = params.get("actor_alpha", 0.05)
        if alpha is not None:
            self.critic_alpha = alpha
            self.actor_alpha = alpha
        self.gamma = params.get("gamma", 0.99)
        self.buffer_max_size = params.get("buffer_max_size", int(1e4))
        self.tau = params.get("tau", 0.005)
        self.layer_1_size = params.get("layer1_size", 64)
        self.layer_2_size = params.get("layer2_size", 64)
        self.batch_size = params.get("batch_size", 64)
        self.reward_scale = params.get("reward_scale", 1)
        self.replay_buffer = ReplayBuffer(self.buffer_max_size, self.device)

        self.policy_update_frequency = 2
        self.learning_step = 1

        self.min_std = -20
        self.max_std = 2

        self.actor = MLP(self.state_size, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                         2 * self.nb_actions, learning_rate=self.actor_lr, optimizer_class=optim.Adam,
                         device=self.device).float()
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = MLP(self.state_size + self.nb_actions, self.layer_1_size, ReLU(),
                          self.layer_2_size, ReLU(), 1, learning_rate=self.critic_lr, optimizer_class=optim.Adam,
                          device=self.device).float()
        self.target_critic = copy.deepcopy(self.critic)
        self.passed_logs = []

        # HER will relabel samples in the last trajectory. To do it, we need to keep this last trajectory in a memory
        self.last_trajectory = []
        # ... and store relabelling parameters
        self.nb_resample_per_states = 4

    def get_q_value(self, state, goal):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device) if isinstance(state, np.ndarray) else state
            goal = torch.from_numpy(goal).to(self.device) if isinstance(goal, np.ndarray) else goal

            if len(state.shape) == 1:
                goal_conditioned_state = torch.cat((state[:self.goal_size] - goal, state[self.goal_size:]), dim=-1)
            elif len(state.shape) == 2:
                goal_conditioned_state = torch.cat((state[:, :self.goal_size] - goal,
                                                    state[:, self.goal_size:]), dim=-1)
            else:
                raise NotImplementedError("Batch shape not supported, not implemented error.")

            next_actions, _ = self.sample_action(goal_conditioned_state, actor_network=self.target_actor)
            critic_input = torch.cat((goal_conditioned_state, next_actions), -1)
            q_values = self.target_critic.forward(critic_input).view(-1)
        return q_values

    def action(self, state):
        with torch.no_grad():
            goal_conditioned_state = np.concatenate((state[:self.goal_size] - self.current_goal, state[self.goal_size:]))
            actions, _ = self.sample_action(goal_conditioned_state)
        return actions.cpu().detach().numpy()

    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            states, actions, rewards, new_states, done, goals = self.replay_buffer.sample(self.batch_size)

            # NEW concatenate states and goals, because we need to put them inside our model
            goal_conditioned_states = np.concatenate((states[:, :self.goal_size] - goals,
                                                      states[:, self.goal_size:]), axis=-1)
            goal_conditioned_states = torch.from_numpy(goal_conditioned_states).to(self.device)
            goal_conditioned_new_states = np.concatenate((new_states[:, :self.goal_size] - goals,
                                                          new_states[:, self.goal_size:]), axis=-1)
            goal_conditioned_new_states = torch.from_numpy(goal_conditioned_new_states).to(self.device)

            # Training critic
            with torch.no_grad():
                next_actions, next_log_probs = \
                    self.sample_action(goal_conditioned_new_states, actor_network=self.target_actor)
                critic_input = torch.cat((goal_conditioned_new_states, next_actions), axis=-1)
                self.passed_logs.append(next_log_probs)
                next_q_values = \
                    self.target_critic.forward(critic_input).view(-1)

            q_hat = self.reward_scale * rewards + self.gamma * (1 - done) * \
                (next_q_values - self.critic_alpha * next_log_probs)
            q_values = self.critic.forward(torch.cat((goal_conditioned_states, actions), axis=-1)).view(-1)
            critic_loss = functional.mse_loss(q_values, q_hat)
            self.critic.learn(critic_loss)
            self.target_critic.converge_to(self.critic, tau=self.tau)

            if self.learning_step % self.policy_update_frequency == 0:
                for _ in range(self.policy_update_frequency):
                    # Train actor
                    actions, log_probs = self.sample_action(goal_conditioned_states)
                    log_probs = log_probs.view(-1)
                    critic_values = self.critic.forward(torch.cat((goal_conditioned_states, actions), axis=-1)).view(-1)

                    actor_loss = self.actor_alpha * log_probs - critic_values
                    actor_loss = torch.mean(actor_loss)
                    self.actor.learn(actor_loss, retain_graph=True)
                    self.target_actor.converge_to(self.actor, tau=self.tau)
            self.learning_step += 1


