import gym
import numpy as np
import random

from dsg_rgl_ant.simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from dsg_rgl_ant.simple_rl.tasks.point_reacher.PointReacherStateClass import PointReacherState
from dsg_rgl_ant.simple_rl.tasks.point_maze.environments.point_maze_env import PointMazeEnv


class D4RLPointMazeMDP(GoalDirectedMDP):
    def __init__(self, difficulty, goal_directed, seed=0, render=False):
        assert difficulty in ("easy", "medium", "hard")

        self.env_name = f"d4rl-{difficulty}-point-maze"
        self.seed = seed
        self.render = render
        self.difficulty = difficulty
        self.goal_directed = goal_directed

        random.seed(seed)
        np.random.seed(seed)

        if difficulty == "easy":
            maze_id = "Maze"
        elif difficulty == "medium":
            maze_id = "d4rl-maze"
        elif difficulty == "hard":
            maze_id = "d4rl-hard-maze"
        else:
            raise NotImplementedError(difficulty)

        # Configure env
        gym_mujoco_kwargs = {
            'maze_id': maze_id,
            'n_bins': 0,
            'observe_blocks': False,
            'put_spin_near_agent': False,
            'top_down_view': False,
            'manual_collision': True,
            'maze_size_scaling': 3,
        }
        self.env = PointMazeEnv(**gym_mujoco_kwargs)
        self.reset()

        self.current_goal = self.env.goal_xy if self.goal_directed else None

        salient_positions = []

        self._determine_x_y_lims()
        self.hard_coded_salient_positions = np.copy(salient_positions)

        GoalDirectedMDP.__init__(self, range(self.env.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func, self.init_state,
                                 salient_positions, task_agnostic=not goal_directed,
                                 goal_state=self.current_goal, goal_tolerance=0.6)

    def set_current_goal(self, goal):
        if self.goal_directed:
            self.current_goal = goal

    def get_current_goal(self):
        if self.goal_directed:
            return self.current_goal
        raise ValueError(f"goal_directed={self.goal_directed}")

    def _determine_salient_positions(self):
        if self.difficulty == "easy":
            salient_positions = [np.array((4, 0)),
                                 np.array((8, 0)),
                                 np.array((8, 4)),
                                 np.array((8, 8)),
                                 np.array((4, 8)),
                                 np.array((0, 8))]
        elif self.difficulty == "medium":
            salient_positions = [np.array((6, 8)),
                                 np.array((5, -5)),
                                 np.array((-7.5, -5)),
                                 np.array((-8.5, 8))]
        elif self.difficulty == "hard":
            salient_positions = [np.array((-15, -10)),
                                 np.array((-15, +10)),
                                 np.array((10, -10)),
                                 np.array((10, 9)),
                                 np.array((-6, +5))]
        else:
            raise NotImplementedError(self.difficulty)

        return salient_positions

    def _reward_func(self, state, action):
        next_state, _, done, info = self.env.step(action)

        if self.task_agnostic:  # No reward function => no rewards and no terminations
            reward = 0.
            is_terminal = False
        else:
            reward, is_terminal = self.sparse_gc_reward_function(next_state, self.get_current_goal(), info)

        if self.render:
            self.env.render()

        self.next_state = self._get_state(next_state, is_terminal)

        return reward

    def _transition_func(self, state, action):
        return self.next_state

    def _get_state(self, observation, done):
        """ Convert np obs array from gym into a State object. """  # TODO: Adapt has_key
        obs = np.copy(observation)
        position = obs[:2]
        has_key = obs[2]
        theta = obs[3]
        velocity = obs[4:6]
        theta_dot = obs[6]
        # Ignoring obs[7] which corresponds to time elapsed in seconds
        state = PointReacherState(position, theta, velocity, theta_dot, done)
        return state

    def execute_agent_action(self, action, option_idx=None):
        reward, next_state = super(D4RLPointMazeMDP, self).execute_agent_action(action)
        return reward, next_state

    @staticmethod
    def state_space_size():
        return 6

    @staticmethod
    def action_space_size():
        return 2

    def reset(self):
        init_state_array = self.env.reset()
        self.init_state = self._get_state(init_state_array, done=False)
        super(D4RLPointMazeMDP, self).reset()

    def set_xy(self, position):
        self.env.wrapped_env.set_xy(position)
        self.cur_state = self._get_state(np.array((position[0], position[1], 0, 0, 0, 0, 0)), done=False)

    def get_init_positions(self):
        return [self.init_state.position]

    def __str__(self):
        return self.env_name

    def _determine_x_y_lims(self):
        if self.difficulty == "easy":
            xlow, xhigh = -2., 10
            ylow, yhigh = -2., 10.
        elif self.difficulty == "medium":
            xlow, xhigh = -10., 7.5
            ylow, yhigh = -7.5, 10.
        elif self.difficulty == "hard":
            xlow, xhigh = -16, +12
            ylow, yhigh = -10, +10
        else:
            raise NotImplementedError(self.difficulty)

        self.xlims = (xlow, xhigh)
        self.ylims = (ylow, yhigh)

    def get_x_y_low_lims(self):
        return self.xlims[0], self.ylims[0]

    def get_x_y_high_lims(self):
        return self.xlims[1], self.ylims[1]

    def sample_random_state(self):
        """ Rejection sampling from the set of feasible states in the maze. """
        default_choices = self._determine_salient_positions()
        num_tries = 0
        rejected = True
        while rejected and num_tries < 200:
            low = np.array((self.xlims[0], self.ylims[0]))
            high = np.array((self.xlims[1], self.ylims[1]))
            sampled_point = np.random.uniform(low=low, high=high)
            rejected = self.env._is_in_collision(sampled_point)
            num_tries += 1

            if not rejected:
                return sampled_point

        return random.choice(default_choices)

    def sample_random_action(self):
        size = (self.action_space_size(),)
        return np.random.uniform(-1., 1., size=size)
