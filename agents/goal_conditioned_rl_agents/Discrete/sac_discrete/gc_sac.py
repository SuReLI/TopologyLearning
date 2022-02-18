import copy
from enum import Enum
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from agents.gc_agent import GoalConditionedAgent
from agents.goal_conditioned_rl_agents.Discrete.discrete_actions_agent import GCAgentDiscrete
# from agents.utils.replay_buffer import ReplayBuffer
from agents.utils.nn.mlp import MLP


class DefaultNN(nn.Module):

    def __init__(self, device, learning_rate, input_size, output_size, layer1_size=256, layer2_size=256, tau=.1):
        super(DefaultNN, self).__init__()
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size
        self.layer1 = nn.Linear(self.input_size, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        self.layer3 = nn.Linear(self.layer2_size, self.output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = device
        self.to(self.device)

        self.tau = tau

    def forward(self, input):
        if isinstance(input, list):
            input = torch.tensor(input).to(self.device, dtype=torch.float32)
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).to(self.device, dtype=torch.float32)
        output = F.relu(self.layer1(input))
        output = F.relu(self.layer2(output))
        output = self.layer3(output)
        return output

    def update_parameters_following(self, other_network, tau=None):
        """
        Fait tendre les paramètres du réseau courant vers les paramètres d'un réseau passé en paramètres.
        :param other_network: Réseau à copier, doit être de la même forme que le réseau courant
        :param tau: Taux de copie, si =0 alors le réseau donné ne sera pas copié, si =1 alors il sera complètement copié
        """
        if tau is None:
            tau = self.tau
        assert isinstance(other_network, DefaultNN)

        current_network_state_dict = dict(self.named_parameters())
        other_network_state_dict = dict(other_network.named_parameters())

        for name in current_network_state_dict:
            current_network_state_dict[name] = tau * other_network_state_dict[name].clone() + \
                    (1 - tau) * current_network_state_dict[name].clone()

        self.load_state_dict(current_network_state_dict)

    def __copy__(self):
        new_one = DefaultNN(self.device, self.learning_rate, self.input_size, self.output_size,
                            layer1_size=self.layer1_size, layer2_size=self.layer2_size, tau=self.tau)
        new_one.update_parameters_following(self, tau=1)
        return new_one


class BufferFillingMode(Enum):
    RESERVOIR_SAMPLING = 1
    LAST_IN_FIRST_OUT = 2


class ReplayBuffer:
    def __init__(self, state_dim, nb_actions, max_size=1000000,
                 filling_mode: BufferFillingMode = BufferFillingMode.LAST_IN_FIRST_OUT):
        self.mem_size = max_size
        self.mem_counter = 0

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, 1))
        self.new_state_memory = np.zeros((self.mem_size, state_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.goal_memory = np.zeros((self.mem_size, state_dim))

        self.is_set = np.zeros(self.mem_size, dtype=bool)
        self.data_index = np.asanyarray([], dtype=np.int64)
        self.filling_mode = filling_mode
        if self.filling_mode == BufferFillingMode.RESERVOIR_SAMPLING:
            self.nb_data_seen = 0

        # Iterator attributes
        self.iterator_pointer = False

    def store_transition(self, state, action, state_, reward, done, goal):
        # Select an index
        if self.filling_mode == BufferFillingMode.LAST_IN_FIRST_OUT:
            index = self.mem_counter % self.mem_size
        elif self.filling_mode == BufferFillingMode.RESERVOIR_SAMPLING:
            if self.nb_data_seen < self.mem_size:
                index = self.nb_data_seen
            else:
                index = randint(0, self.nb_data_seen)
                if index >= self.mem_size:
                    return
                # else : insert the data inside the memory at index randomly sampled
                # NB: this reservoir sampling method allow each seen data to have the same probability to be inside the
                # buffer at any time.
            self.nb_data_seen += 1
        else:
            raise Exception("Unknown filling method.")
        if not self.is_set[index]:
            self.data_index = np.append(self.data_index, index)
            self.is_set[index] = True

        # Insert the data at the selected index
        self.state_memory[index] = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state
        self.action_memory[index] = action
        self.new_state_memory[index] = state_.detach().cpu().numpy() if isinstance(state_, torch.Tensor) else state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.goal_memory[index] = goal.detach().cpu().numpy() if isinstance(goal, torch.Tensor) else goal

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        # print("self.data_index = " + str(self.data_index))
        # print("batch_size = " + str(batch_size))
        batch_index = np.random.choice(self.data_index, batch_size)

        # print("batch_index = " + str(batch_index))
        for index in self.data_index:
            assert self.is_set[index]

        states = self.state_memory[batch_index]
        actions = self.action_memory[batch_index]
        states_ = self.new_state_memory[batch_index]
        rewards = self.reward_memory[batch_index]
        done = self.terminal_memory[batch_index]
        goals = self.goal_memory[batch_index]

        return states, actions, states_, rewards, done, goals

    def __iter__(self):
        self.iterator_pointer = 0
        return self

    def __next__(self) -> tuple:
        if self.iterator_pointer < len(self.data_index):
            self.iterator_pointer += 1
            return \
                self.state_memory[self.data_index[self.iterator_pointer - 1]], \
                self.action_memory[self.data_index[self.iterator_pointer - 1]], \
                self.new_state_memory[self.data_index[self.iterator_pointer - 1]], \
                self.reward_memory[self.data_index[self.iterator_pointer - 1]], \
                self.terminal_memory[self.data_index[self.iterator_pointer - 1]],
        else:
            raise StopIteration

    def is_full(self):
        return self.mem_counter >= self.mem_size


class GCSACAgentDiscrete(GoalConditionedAgent):
    def __init__(self, state_space, action_space, device, name="SAC",
                 actor_lr=0.0005, critic_lr=0.0005, discount_factor=0.98,
                 max_buffer_size=3000, tau=0.005, layer1_size=300, layer2_size=300, batch_size=250, sac_temperature=.1):
        """
        This class implement a soft actor critic (SAC) with goal conditioned policy.
         - state_space: Environment state space (gim.spaces.Box)
         - action_space: Environment action space (gim.spaces.Discrete)
         - device: The device where the NN should run on.
         - actor_lr: Learning rate of the SAC actor.
         - critic_lr: Learning rate of the SAC critic.
         - gamma: Discount factor.
         - max_buffer_size: Maximum size of the buffer.
         - tau: Parameter that indicate how much target networks should reach main ones (0 < tau <= 1).
         - layer1_size: size of networks first layers (each networks have the same size for simplicity).
         - layer2_size: size of networks second layers.
         - batch_size: Learning batch size (same for actor, critic, and VAE
         - alpha: SAC temperature.
        """
        super().__init__(state_space, action_space, device, name=name)
        self.current_goal = None
        self.gamma = discount_factor
        self.tau = tau
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.worst_intrinsic_reward = 0
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor

        self.interactions_memory = ReplayBuffer(self.state_size, self.nb_actions,
                                                max_size=self.max_buffer_size)

        # Soft Actor Critic NNs
        self.actor = DefaultNN(self.device, actor_lr, self.state_size * 2, action_space.n,
                               layer1_size=layer1_size, layer2_size=layer2_size)
        self.critic_1 = DefaultNN(self.device, critic_lr, self.state_size * 2, action_space.n,
                                  layer1_size=layer1_size, layer2_size=layer2_size)
        self.critic_2 = DefaultNN(self.device, critic_lr, self.state_size * 2, action_space.n,
                                  layer1_size=layer1_size, layer2_size=layer2_size)

        # Soft Actor Critic attributes
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.sac_temperature = sac_temperature

    def learn(self):
        """
        Make the entire algorithm learn.
        """

        states, actions, next_states, rewards, done, goals = self.interactions_memory.sample_buffer(self.batch_size)

        states = torch.from_numpy(states).to(dtype=torch.float32, device=self.device)
        actions = torch.from_numpy(actions).to(dtype=torch.int64, device=self.device)
        next_states = torch.from_numpy(next_states).to(dtype=torch.float32, device=self.device)
        rewards = torch.from_numpy(rewards).to(dtype=torch.float32, device=self.device)
        done = torch.from_numpy(done).to(dtype=torch.float32, device=self.device)
        goals = torch.from_numpy(goals).to(dtype=torch.float32, device=self.device)

        goal_conditioned_states = torch.concat((states, goals), dim=-1)
        goal_conditioned_next_states = torch.concat((next_states, goals), dim=-1)

        self.train_critic(goal_conditioned_states, actions, goal_conditioned_next_states, rewards, done)
        self.train_actor(goal_conditioned_states, actions, goal_conditioned_next_states, rewards, done)

    def train_critic(self, states, actions, next_states, rewards, done):
        # Train critic: Q-target computation
        with torch.no_grad():
            _, next_probabilities = self.sample_action(next_states)
            next_log_probs = torch.log(next_probabilities)
            next_target_q1_preds = self.target_critic_1.forward(next_states)
            next_target_q2_preds = self.target_critic_2.forward(next_states)
            next_target_q_preds = torch.min(next_target_q1_preds, next_target_q2_preds)

            next_v = (next_probabilities * (next_target_q_preds - self.sac_temperature * next_log_probs)).sum(-1)
            target_q = rewards + self.gamma * (1 - done) * next_v

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_old_policy = self.critic_1.forward(states).gather(1, actions).view(-1)
        q2_old_policy = self.critic_2.forward(states).gather(1, actions).view(-1)
        critic_1_loss = F.mse_loss(q1_old_policy, target_q)
        critic_2_loss = F.mse_loss(q2_old_policy, target_q)
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Correct targets
        self.target_critic_1.update_parameters_following(self.critic_1, tau=self.tau)
        self.target_critic_2.update_parameters_following(self.critic_2, tau=self.tau)

    def train_actor(self, states, actions, next_states, rewards, done):
        # Train actor
        _, probabilities = self.sample_action(states, actor_network=self.actor)
        log_probs = torch.log(probabilities)
        with torch.no_grad():
            q1 = self.critic_1.forward(states)
            q2 = self.critic_2.forward(states)
            q_value = torch.min(q1, q2)

        policy_loss = (probabilities * (self.sac_temperature * log_probs - q_value)).sum(-1).mean()
        self.actor.optimizer.zero_grad()
        policy_loss.backward()
        self.actor.optimizer.step()
        self.target_actor.update_parameters_following(self.actor, tau=self.tau)

    def sample_action(self, states_batch, actor_network=None):
        if len(states_batch.shape) == 1:
            states_batch = torch.from_numpy(states_batch).to(self.device, dtype=torch.float32)
            goal = torch.from_numpy(self.current_goal).to(self.device, dtype=torch.float32)
            states_batch = torch.concat((states_batch, goal), -1)

        if actor_network is None:
            actor_network = self.target_actor
        assert isinstance(actor_network, DefaultNN)
        out_probabilities = actor_network.forward(states_batch)
        probabilities = torch.nn.functional.softmax(out_probabilities, -1)  # Make sure it works on batch
        z = probabilities == 0.0
        z = z.float() * 1e-8
        distribution = torch.distributions.Categorical(probabilities)
        return distribution.sample(), probabilities + z

    def action(self, state):
        """
        Return an action chosen from the given state
        Precondition: State is A SINGLE state and NOT A BATCH, use sample_action instead
        """
        action, _ = self.sample_action(state)
        return action.item()

    def on_action_stop(self, action, new_state, reward, done) -> float:
        # Store transition into the trajectory memory. We do not want goals inside it since those states are used for
        # relabelling only.

        self.interactions_memory.store_transition(self.last_state, action, new_state, reward, done, self.current_goal)
        self.learn()
        return super().on_action_stop(action, new_state, reward, done)

    def reset_buffer(self, state_size=None):
        if state_size is None:
            state_size = self.state_size
        self.interactions_memory = ReplayBuffer(state_size, self.nb_actions, max_size=self.max_buffer_size)

    def reset(self):
        self.__init__(state_space=self.state_space, action_space=self.action_space, device=self.device, name=self.name,
                      actor_lr=self.actor_lr, critic_lr=self.critic_lr, discount_factor=self.discount_factor,
                      max_buffer_size=self.max_buffer_size, tau=self.tau, layer1_size=self.layer1_size,
                      layer2_size=self.layer2_size, batch_size=self.batch_size, sac_temperature=self.sac_temperature)
