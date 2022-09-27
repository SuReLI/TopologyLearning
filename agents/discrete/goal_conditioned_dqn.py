# Goal conditioned deep Q-network

import copy
import numpy as np
import torch
from torch import optim
from torch.nn import ReLU

from agents.gc_agent import GoalConditionedAgent
from agents.utils.mlp import MLP
from agents.utils.replay_buffer import ReplayBuffer


class GCDQNAgent(GoalConditionedAgent):
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
        params["name"] = params.get("name", "DQN")  # Set if not already set
        super().__init__(**params)

        self.gamma = params.get("gamma", 0.95)
        self.epsilon_min = params.get("epsilon_min", 0.01)
        self.epsilon_max = params.get("epsilon_max", 1.)
        self.epsilon_decay_delay = params.get("epsilon_decay_delay", 20)
        self.epsilon = None
        self.epsilon_decay_period = params.get("epsilon_decay_period", 1000)
        self.buffer_size = params.get("buffer_size", 1000000)
        self.layer_1_size = params.get("layer_1_size", 250)
        self.layer_2_size = params.get("layer_2_size", 200)

        #  NEW, goals will be stored inside the replay buffer. We need a specific one with enough place to do so
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.batch_size = params.get("batch_size", 125)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.tau = params.get("tau", 0.001)
        self.nb_gradient_steps = params.get("nb_gradient_steps", 1)

        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_period
        self.total_steps = 0

        # NEW, The input observation size is multiplied by two because we need to also take the goal as input
        self.model = MLP(self.state_size * 2, self.layer_1_size, ReLU(), self.layer_2_size, ReLU(), self.nb_actions,
                         learning_rate=self.learning_rate, optimizer_class=optim.Adam, device=self.device).float()

        self.criterion = torch.nn.SmoothL1Loss()
        self.target_model = copy.deepcopy(self.model).to(self.device)

    def get_q_value(self, state, goal):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device) if isinstance(state, np.ndarray) else state
            goal = torch.from_numpy(goal).to(self.device) if isinstance(goal, np.ndarray) else goal
            goal_conditioned_state = torch.cat((state, goal), dim=-1)

            if len(state.shape) == 1:
                return torch.max(self.model(goal_conditioned_state)).detach().item()
            elif len(state.shape) == 2:
                return torch.max(self.model(goal_conditioned_state), dim=-1).values
            else:
                raise NotImplementedError("Batch shape not supported, not implemented error.")

    def on_simulation_start(self):
        self.epsilon = self.epsilon_max
        super().on_simulation_start()

    def on_episode_start(self, *args):
        return super().on_episode_start(*args)

    def action(self, state):
        if self.simulation_time_step_id > self.epsilon_decay_delay:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        if np.random.rand() < self.epsilon:  # Epsilon greedy
            action = np.random.randint(self.nb_actions)
        else:
            # greedy_action(self.model, observation) function in RL5 notebook
            with torch.no_grad():
                q_values = self.model(np.concatenate((state, self.current_goal)))
                action = torch.argmax(q_values).item()
        return action

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        if learn:
            assert isinstance(new_state, np.ndarray)
            assert reward is not None
            self.replay_buffer.append((self.last_state, action, reward, new_state, done, self.current_goal))
            self.learn()
        super().on_action_stop(action, new_state, reward, done, learn=learn)  # Replace self.last_state by the new_state

    def learn(self):
        for _ in range(self.nb_gradient_steps):
            # gradient_step() function in RL5 notebook
            if len(self.replay_buffer) > self.batch_size:
                #  NEW, samples from buffer contains goals
                states, actions, rewards, new_states, dones, goals = \
                    self.replay_buffer.sample(self.batch_size)

                # NEW concatenate states and goals, because we need to put them inside our model
                goal_conditioned_states = np.concatenate((states, goals), -1)
                goal_conditioned_new_states = np.concatenate((new_states, goals), -1)

                q_prime = self.target_model(goal_conditioned_new_states).max(1)[0].detach()
                update = rewards + self.gamma * (1 - dones) * q_prime
                q_s_a = self.model(goal_conditioned_states).gather(1, actions.to(torch.long).unsqueeze(1))
                loss = self.criterion(q_s_a, update.unsqueeze(1))
                self.model.learn(loss)

        self.target_model.converge_to(self.model, self.tau)