from random import randrange

from agents.goal_conditioned_rl_agents.Discrete.sac_discrete.gc_sac import GCSACAgentDiscrete


class SACHERAgent(GCSACAgentDiscrete):
    """
    A DQN agent that use HER.
    """

    def __init__(self, state_space, action_space, device, name="SAC", actor_lr=0.0005, critic_lr=0.0005,
                 discount_factor=0.98, max_buffer_size=100000, tau=0.005, layer1_size=125, layer2_size=100,
                 batch_size=150, sac_temperature=.1):

        super().__init__(
            state_space=state_space, action_space=action_space, device=device, name=name, actor_lr=actor_lr,
            critic_lr=critic_lr, discount_factor=discount_factor, max_buffer_size=max_buffer_size, tau=tau,
            layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, sac_temperature=sac_temperature
        )

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
                reward = new_state_index / goal_index
                self.interactions_memory.store_transition(state, action, new_state, reward,
                                                          goal_index == new_state_index, goal)
                # self.replay_buffer.append((state, action, reward, new_state, goal_index == new_state_index, goal))

    def reset(self):
        self.__init__(state_space=self.state_space, action_space=self.action_space, device=self.device, name=self.name,
                      actor_lr=self.actor_lr, critic_lr=self.critic_lr, discount_factor=self.discount_factor,
                      max_buffer_size=self.max_buffer_size, tau=self.tau, layer1_size=self.layer1_size,
                      layer2_size=self.layer2_size, batch_size=self.batch_size, sac_temperature=self.sac_temperature)
