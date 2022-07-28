import math

import numpy as np

from src.environments.grid_world import DiscreteGridWorld
import random
from src.environments.grid_world import settings

from src.environments.grid_world.utils.indexes import Colors


class GoalConditionedDiscreteGridWorld(DiscreteGridWorld):
    def __init__(self, map_id=settings.map_id, stochasticity=0.):
        super().__init__(map_id=map_id, stochasticity=stochasticity)
        self.goal_coordinates = None
        self.goal = None
        self.reset_goal()

    def reset_goal(self):
        """
        Choose a goal for the agent.
        """
        oracle = self.get_oracle(coordinates=True)  # Free of unreachable states
        self.goal_coordinates = random.choice(oracle)
        self.goal = self.get_state(*self.goal_coordinates)
        return self.goal

    def goal_reached(self, distance=0, goal=None):
        """
        Return a boolean True if the agent observation is on the goal (and exactly on the goal since our observation space is
        graph_free here in reality), and false otherwise.
        Parameter 'distance' indicate the tolerance radius allowed between the current observation and the goal to be
            considered as reached.
        """
        x, y = self.agent_coordinates
        if isinstance(goal, np.ndarray):
            g_x, g_y = self.get_coordinates(goal)
        else:
            if goal is None:
                goal = self.goal_coordinates
            g_x, g_y = goal

        return distance >= math.sqrt((x - g_x)**2 + (y - g_y)**2)

    def step(self, action, goal=None):
        """
        Compute a stem in the MDP loop.
        :param action: The action executed by the agent.
        :param goal: If the goal the agent is trying to reach is different from the one the environment sampled.
        :return: new observation, reward, done
        """
        new_x, new_y = self.get_new_coordinates(action)
        if self.is_available(new_x, new_y):
            self.agent_coordinates = new_x, new_y

            # Compute goal coordinates
            if goal is None:
                goal = self.goal_coordinates
            elif isinstance(goal, np.ndarray):
                goal = self.get_coordinates(goal)

            done = self.goal_reached(goal=goal)
            reward = -1 if not done else 1
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), reward, done
        else:
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), -1, False

    def reset(self) -> tuple:
        """
        Return the initial observation, and the selected goal.
        """
        self.reset_goal()
        self.agent_coordinates = self.start_coordinates
        return self.get_state(*self.agent_coordinates), self.goal

    def render(self, mode='human'):
        """
        Render the whole-grid human view (get view from super class then add the goal over the image)
        """
        img = super().render(mode=mode)
        goal_x, goal_y = self.goal_coordinates
        return self.set_tile_color(img, goal_x, goal_y, Colors.GOAL.value)
