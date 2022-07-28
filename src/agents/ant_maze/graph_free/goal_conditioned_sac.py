# Goal conditioned deep Q-network

import copy

import numpy as np
import torch
from torch.nn import ReLU
from torch.nn.functional import normalize

from src.agents.utils.mlp import MLP
from src.agents.utils.replay_buffer import ReplayBuffer
from src.agents.gc_agent import GoalConditionedAgent
from torch import optim
from torch.nn import functional
from torch.distributions.normal import Normal


class GoalConditionedSACAgent(GoalConditionedAgent):
    def __init__(self, **params):
        params["name"] = params.get("name", "SAC")
        super().__init__(**params)
        """
        self.actor_lr = params.get("actor_lr", 0.0005)
        self.critic_lr = params.get("critic_lr", 0.0005)
        alpha = params.get("alpha", None)
        self.critic_alpha = params.get("critic_alpha", 0.15)
        self.actor_alpha = params.get("actor_alpha", 0.15)
        if alpha is not None:
            self.critic_alpha = alpha
            self.actor_alpha = alpha
        self.gamma = params.get("gamma", 0.99)
        self.buffer_max_size = params.get("buffer_max_size", int(1e4))
        self.tau = params.get("tau", 0.005)
        self.layer_1_size = params.get("layer1_size", 120)
        self.layer_2_size = params.get("layer2_size", 64)
        self.batch_size = params.get("batch_size", 250)
        self.reward_scale = params.get("reward_scale", 15)
        self.replay_buffer = ReplayBuffer(self.buffer_max_size, self.device)
        """
        self.actor_lr = params.get("actor_lr", 0.0005)
        self.critic_lr = params.get("critic_lr", 0.0005)
        alpha = params.get("alpha", None)
        self.critic_alpha = params.get("critic_alpha", 0.6)
        self.actor_alpha = params.get("actor_alpha", 0.6)
        if alpha is not None:
            self.critic_alpha = alpha
            self.actor_alpha = alpha
        self.gamma = params.get("gamma", 0.99)
        self.buffer_max_size = params.get("buffer_max_size", int(1e4))
        self.tau = params.get("tau", 0.005)
        self.layer_1_size = params.get("layer1_size", 250)
        self.layer_2_size = params.get("layer2_size", 150)
        self.batch_size = params.get("batch_size", 250)
        self.reward_scale = params.get("reward_scale", 15)
        self.replay_buffer = ReplayBuffer(self.buffer_max_size, self.device)

        self.policy_update_frequency = 2
        self.learning_step = 1

        self.min_std = -20
        self.max_std = 2

        self.actor = MLP(self.state_size * 2, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(), 2 * self.nb_actions,
                         learning_rate=self.actor_lr, optimizer_class=optim.Adam, device=self.device).float()
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = MLP(self.state_size * 2 + self.nb_actions, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(),
                          1, learning_rate=self.critic_lr, optimizer_class=optim.Adam, device=self.device).float()
        self.target_critic = copy.deepcopy(self.critic)

        self.passed_logs = []

    def get_q_value(self, state, goal):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device) if isinstance(state, np.ndarray) else state
            goal = torch.from_numpy(goal).to(self.device) if isinstance(goal, np.ndarray) else goal
            goal_conditioned_state = torch.cat((state, goal), dim=-1)

            next_actions, _ = self.sample_action(goal_conditioned_state, actor_network=self.target_actor)
            critic_input = torch.cat((goal_conditioned_state, next_actions), -1)
            q_values = self.target_critic.forward(critic_input).view(-1)
        return q_values

    def sample_action(self, actor_input, actor_network=None):
        if actor_network is None:
            actor_network = self.actor

        if isinstance(actor_input, np.ndarray):
            actor_input = torch.from_numpy(actor_input).to(self.device)
        actor_input = normalize(actor_input, p=2., dim=-1)

        # Forward
        actor_output = actor_network(actor_input)
        if len(actor_input.shape) > 1:  # It's a batch
            actions_means = actor_output[:, :self.nb_actions]
            actions_log_stds = actor_output[:, self.nb_actions:]
        else:
            actions_means = actor_output[:self.nb_actions]
            actions_log_stds = actor_output[self.nb_actions:]

        actions_log_stds = torch.clamp(actions_log_stds, min=self.min_std, max=self.max_std)
        actions_stds = torch.exp(actions_log_stds)
        actions_distribution = Normal(actions_means, actions_stds)

        raw_actions = actions_distribution.rsample()

        log_probs = actions_distribution.log_prob(raw_actions)
        actions = torch.tanh(raw_actions)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(-1)
        return actions, log_probs

    def action(self, state):
        with torch.no_grad():
            actor_input = np.concatenate((state, self.current_goal))
            actions, _ = self.sample_action(actor_input)
        return actions.cpu().detach().numpy()

    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            states, actions, rewards, new_states, done, goals = self.replay_buffer.sample(self.batch_size)

            # NEW concatenate states and goals, because we need to put them inside our model
            goal_conditioned_states = torch.cat((states, goals), dim=-1)
            goal_conditioned_new_states = torch.cat((new_states, goals), dim=-1)

            # Training critic
            with torch.no_grad():
                next_actions, next_log_probs = \
                    self.sample_action(goal_conditioned_new_states, actor_network=self.target_actor)
                critic_input = torch.cat((goal_conditioned_new_states, next_actions), -1)
                self.passed_logs.append(next_log_probs)
                next_q_values = \
                    self.target_critic.forward(critic_input).view(-1)

            q_hat = self.reward_scale * rewards + self.gamma * (1 - done) * \
                (next_q_values - self.critic_alpha * next_log_probs)
            q_values = self.critic.forward(torch.cat((goal_conditioned_states, actions), -1)).view(-1)
            critic_loss = functional.mse_loss(q_values, q_hat)
            self.critic.learn(critic_loss)
            self.target_critic.converge_to(self.critic, tau=self.tau)

            if self.learning_step % self.policy_update_frequency == 0:
                for _ in range(self.policy_update_frequency):
                    # Train actor
                    actions, log_probs = self.sample_action(goal_conditioned_states)
                    log_probs = log_probs.view(-1)
                    critic_values = self.critic.forward(torch.cat((goal_conditioned_states, actions), -1)).view(-1)

                    actor_loss = self.actor_alpha * log_probs - critic_values
                    actor_loss = torch.mean(actor_loss)
                    self.actor.learn(actor_loss, retain_graph=True)
                    self.target_actor(self.actor, tau=self.tau)
            self.learning_step += 1

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        if learn:
            self.replay_buffer.append((self.last_state, action, reward, new_state, done, self.current_goal))
            self.learn()
        super().on_action_stop(action, new_state, reward, done, learn=learn)  # Replace self.last_state by the new_state
