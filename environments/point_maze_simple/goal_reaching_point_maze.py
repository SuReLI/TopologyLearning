from typing import Tuple

import numpy as np
from old.src.environments.point_maze_simple.point_maze import PointEnv
from ..environment import GoalReachingEnvironment


class GoalReachingPointEnv(GoalReachingEnvironment, PointEnv):
    def __init__(self, maze_name, action_noise=1., reachability_threshold=1, **rendering_options):
        self.goal = None
        PointEnv.__init__(self, maze_name, action_noise, **rendering_options)

        self.reachability_threshold = reachability_threshold
        self.goal_space = self.observation_space.copy()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        state = super().reset()
        self.goal = self._sample_empty_state()
        return state, self._normalize_obs(self.goal.copy())

    def step(self, action) -> Tuple[np.ndarray, float, bool]:
        new_state, _, done = PointEnv.step(self, action)

        reached = np.linalg.norm(self.state - self.goal, 2) < self.reachability_threshold
        done |= reached
        reward = 0. if reached else -1
        return self._normalize_obs(self.state.copy()), reward, done

    def extract_goal(self, state) -> np.ndarrray:
        return state.copy

    def render(self) -> np.ndarray:
