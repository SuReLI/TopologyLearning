import random
import numpy as np
from gym.spaces import Box, Discrete
from ...goal_conditioned_wrappers.goal_conditioned_agent import GoalConditionedAgent
from ...goal_conditioned_wrappers.goal_conditioned_value_based_agent import GoalConditionedValueBasedAgent

"""
This implementation of Hierarchical Actor-Critic (HAC) use hindsight experience replay to learn. It is an implementation
of the paper "Learning multi-level hierarchies with hindsight" from Andrew Levy et al. (ICLR 2019).
"""

class HRL(GoalConditionedAgent):
    def __init__(self, control_algorithm, nb_layers=3, high_level_agent=None, **params):
        """
        @param control_algorithm: The RL algorithm used to select the lowest level action (aka. control action).
        @param nb_layers: how many layers do we use. Should be > 1.
        @param high_level_agent: The RL algorithm used to generate sub_goals. If None, then we copy the
            control policy to get the sub-goals generator algorithm.

        NB: The sub-goal generation class can be different from the control policy because if the control action space
        is different from the raw state spaces (the ones of the lowest level) then the action space of the higher layer
        can be continuous while the action space of the lowest is discrete (for example).

        @param params: Optional parameters (check __init__ function for more details).
        """
        # Run some tests about parameters and set action_space and state_space using given agents.
        assert isinstance(control_algorithm, GoalConditionedValueBasedAgent)

        state_space = control_algorithm.state_space
        action_space = control_algorithm.action_space
        assert state_space == high_level_agent.state_space

        # Initialise super class and then initialise us.
        super().__init__(state_space, action_space, **params)

        self.name = "HRL(" + control_algorithm.name + ", " + str(nb_layers) \
                    + ("" if high_level_agent is None else ", " + high_level_agent.name) + ")"

        # Compute sub_goal space
        self.sub_goal_space = params.get("goal_space", self.state_space)
        self.sub_goal_size = self.sub_goal_space.shape[0]
        self.sub_goal_shape = self.sub_goal_space.shape
        assert isinstance(self.sub_goal_space, Box) or isinstance(self.sub_goal_space, Discrete)
        assert len(self.sub_goal_shape) == 1, "Multi dimensional spaces are not supported."

        # Store arguments in memory (params are stored by the mother class)
        self.control_algorithm = control_algorithm
        self.high_level_agent = control_algorithm.copy() if high_level_agent is None else high_level_agent.copy()
        assert isinstance(high_level_agent, GoalConditionedValueBasedAgent)
        self.high_level_agent.action_space = self.sub_goal_space
        self.high_level_agent.reset()
        # NB: other arguments are stored by the mother class.

        # Compute state_to_sub_goal_filter, (state[self.state_to_sub_goal_filter] = <goal associated to state>) aka a projection
        default_filter = np.array([1] * self.sub_goal_size + [0] * (self.state_size - self.sub_goal_size)).astype(bool)
        self.state_to_sub_goal_filter = params.get("state_to_sub_goal_filter", default_filter)
        assert np.argwhere(self.state_to_sub_goal_filter).shape[0] == self.sub_goal_size

        self.planning_horizon = params.get("planning_horizon", 20)
        self.sg_test_probability = params.get("sub_goal_test_probability", 0)
        self.layer_id = nb_layers - 1

        self.reachability_threshold = params.get("reachability_threshold", .8)

        # Layers initialisations
        if self.layer_id == 1:
            self.control_policy_trials_to_reach_sg = 0
        self.testing_sub_goal = False
        self.last_trajectory = []

        if self.layer_id == 1:
            self.lower_level = self.control_algorithm
        else:
            self.lower_level = HRL(control_algorithm, nb_layers - 1, high_level_agent, **params)

        self.done = False
        self.sub_goal = None
        self.nb_sub_goals_sampled = 0

        self.use_basic_experience_replay = True
        self.use_sub_goals_penalty = False
        self.use_hindsight_actions_replay = False
        self.use_hindsight_goal_replays = True

    @property
    def sub_goals(self):
        if self.layer_id == 1:
            return [self.sub_goal, self.current_goal]
        else:
            return self.lower_level.sub_goals + [self.current_goal]

    def get_layer(self, layer_id):
        assert layer_id >= 0
        if self.layer_id == layer_id:
            return self
        if (self.layer_id - 1) == layer_id:
            return self.lower_level
        else:
            return self.lower_level.get_layer(layer_id)

    def start_episode(self, state: np.ndarray, goal: np.ndarray, test_episode=False):
        super().start_episode(state, goal, test_episode)
        self.testing_sub_goal = False
        self.done = False
        self.nb_sub_goals_sampled = 0
        self.last_trajectory = []

        self.high_level_agent.start_episode(state, goal, test_episode)
        self.reset_sub_goal(state)

    def process_interaction(self, action, reward, new_state, done, learn=True):

        reached = np.linalg.norm(new_state - self.sub_goal) < self.reachability_threshold
        self.lower_level.process_interaction(action, 0 if reached else -1, new_state, reached or done, learn)

        if reached or (self.layer_id == 1 and self.control_policy_trials_to_reach_sg >= self.planning_horizon) \
                or (self.layer_id > 1 and self.lower_level.done):
            action = self.sub_goal
            super().process_interaction(action, reward, new_state, done)
            if learn and not self.under_test:
                self.last_trajectory.append((self.last_state, action))
                self.high_level_agent.process_interaction(action, reward, new_state, done,
                                                          learn=self.use_basic_experience_replay)
                if not self.use_basic_experience_replay:
                    self.high_level_agent.learn()

                if self.use_sub_goals_penalty:
                    if not reached and self.testing_sub_goal:
                        if np.linalg.norm(self.last_state - self.sub_goal) < 4:
                            a = 1
                        self.high_level_agent.save_interaction(self.last_state, action, -self.planning_horizon,
                                                               new_state, True)
            if self.nb_sub_goals_sampled >= self.planning_horizon:
                self.done = True
            self.reset_sub_goal(new_state)

        # sub-goals verification routine
        if self.sub_goal is None:
            print("goal is None")
        assert (self.lower_level.current_goal == self.sub_goal).all()

    def action(self, state):
        if self.layer_id == 1:
            self.control_policy_trials_to_reach_sg += 1
        return self.lower_level.action(state)

    def stop_episode(self):
        super().stop_episode()

        # For each observation seen :
        if self.use_hindsight_goal_replays or self.use_hindsight_actions_replay:
            for state_index, (state, action) in enumerate(self.last_trajectory[:-1]):
                new_state_index = state_index + 1
                new_state, _ = self.last_trajectory[new_state_index]

                hindsight_goal_index = len(self.last_trajectory) - 1
                # reward = (new_state_index / goal_index) - 1  # For a more dense reward
                reward = 0 if new_state_index == hindsight_goal_index else -1
                done = hindsight_goal_index == new_state_index

                # sample four goals in future states
                if self.use_hindsight_actions_replay:
                    hindsight_action = new_state.copy()
                    self.high_level_agent.save_interaction(state, hindsight_action, reward, new_state, done,
                                                           self.current_goal)

                if self.use_hindsight_goal_replays:
                    hindsight_goal, _ = self.last_trajectory[-1]
                    hindsight_goal = hindsight_goal[self.state_to_goal_filter]
                    self.high_level_agent.save_interaction(state, action, reward, new_state, done, hindsight_goal)

        self.high_level_agent.stop_episode()
        self.lower_level.stop_episode()

    def reset_sub_goal(self, state):
        assert self.current_goal is not None
        assert (self.current_goal == self.high_level_agent.current_goal).all()

        self.sub_goal = self.high_level_agent.action(state)
        self.nb_sub_goals_sampled += 1
        self.testing_sub_goal = self.sg_test_probability > random.random()
        if self.layer_id == 1:
            self.control_policy_trials_to_reach_sg = 0

        # Reset low level agent's episode
        self.lower_level.start_episode(state, self.sub_goal, test_episode=self.under_test)
        if isinstance(self.lower_level, HRL):
            self.lower_level.testing_sub_goal = self.testing_sub_goal

    def reset(self):
        self.control_algorithm.reset()
        self.high_level_agent.reset()
        self.__init__(self.control_algorithm, self.layer_id + 1, self.high_level_agent, **self.init_params)