# Goal conditioned soft actor-critic
import copy
from statistics import mean

import numpy as np
import torch
from torch import optim
import torch.nn.functional as f
from torch.distributions.normal import Normal
from torch.nn import ReLU

from agents.gc_agent import GoalConditionedAgent
from agents.utils.nn.mlp import MLP
from agents.utils.replay_buffer import ReplayBuffer


class GCSACAgent(GoalConditionedAgent):
    def __init__(self, state_space, action_space, device, actor_lr=0.001, critic_lr=0.001, gamma=0.98,
                 buffer_max_size=10000, tau=0.005, layer_1_size=128, layer_2_size=128, batch_size=128, alpha=1):
        super().__init__(state_space, action_space, device, "SAC")
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_max_size, self.device)
        self.batch_size = batch_size
        self.actions_bounds_range = torch.tensor((action_space.high[0] - action_space.low[0]) / 2).to(self.device)
        self.actions_bounds_mean = torch.tensor(mean((action_space.high[0], action_space.low[0]))).to(self.device)
        self.min_std = 1e-6
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.buffer_max_size = buffer_max_size
        self.layer_1_size = layer_1_size
        self.layer_2_size = layer_2_size

        self.actor = MLP(self.state_size, layer_1_size, ReLU(), layer_2_size, ReLU(), 2 * self.nb_actions,
                         device=self.device, learning_rate=actor_lr, optimizer_class=optim.Adam).float()
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = MLP(self.state_size + self.nb_actions, layer_1_size, ReLU(), layer_2_size, ReLU(), 1,
                          device=device, learning_rate=critic_lr, optimizer_class=optim.Adam).float()
        self.target_critic = copy.deepcopy(self.critic)

        self.alpha = alpha

    def sample_action(self, state, reparameterize=False, actor_network=None):
        if actor_network is None:
            actor_network = self.actor

        # Forward
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        assert isinstance(state, torch.Tensor)
        gc_state = torch.concat((state, self.current_goal), dim=-1)
        actor_output = actor_network(gc_state)

        if len(gc_state.shape) > 1:  # It's a batch
            actions_means = actor_output[:, :self.nb_actions]
            actions_stds = actor_output[:, self.nb_actions:]
        else:
            actions_means = actor_output[:self.nb_actions]
            actions_stds = actor_output[self.nb_actions:]

        actions_stds = torch.clamp(actions_stds, min=self.min_std, max=1)
        actions_distribution = Normal(actions_means, actions_stds)

        if reparameterize:
            actions = actions_distribution.rsample()
        else:
            actions = actions_distribution.sample()

        action = torch.tanh(actions) * self.actions_bounds_range + self.actions_bounds_mean
        log_probs = actions_distribution.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.min_std)
        log_probs = log_probs.sum(dim=-1)

        return action, log_probs

    def action(self, state):
        actions, _ = self.sample_action(state, reparameterize=False)

        return actions.cpu().detach().numpy()

    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            states, actions, rewards, next_states, done, goals = self.replay_buffer.sample(self.batch_size)

            gc_state = torch.cat((states, goals), dim=-1)
            gc_next_state: torch.Tensor = torch.cat((next_states, goals), dim=-1)

            # Training critic
            with torch.no_grad():
                next_actions, next_log_probs = self.sample_action(gc_next_state, actor_network=self.target_actor)
                next_q_values = self.target_critic.forward(torch.cat((gc_next_state, next_actions), -1)).view(-1)
            q_hat = rewards + self.gamma * (1 - done) * (next_q_values - self.alpha * next_log_probs)
            q_values = self.critic(torch.cat((gc_state, actions), 1)).view(-1)
            self.critic.learn(f.mse_loss(q_values, q_hat))
            self.target_critic.converge_to(self.critic, tau=self.tau)

            # Train actor
            actions, log_probs = self.sample_action(gc_state, reparameterize=True)
            log_probs = log_probs.view(-1)
            critic_values = self.critic(torch.cat((gc_state, actions), -1)).view(-1)
            self.actor.learn(torch.mean(log_probs - critic_values))
            self.target_actor.converge_to(self.actor, tau=self.tau)

    def on_action_stop(self, action, new_state, reward, done):
        self.replay_buffer.append((self.last_state, action, reward, new_state, done))
        self.learn()
        super().on_action_stop(action, new_state, reward, done)

    def reset(self):
        self.__init__(self.state_space, self.action_space, self.device, self.name, actor_lr=self.actor_lr,
                      critic_lr=self.critic_lr, gamma=self.gamma, buffer_max_size=self.buffer_max_size, tau=self.tau,
                      layer_1_size=self.layer_1_size, layer_2_size=self.layer_2_size, batch_size=self.batch_size,
                      alpha=self.alpha)
