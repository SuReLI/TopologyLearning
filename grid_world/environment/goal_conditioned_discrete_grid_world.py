import random

import numpy as np

from .discrete_grid_world import DiscreteGridWorld
from .maps.maps_index import MapsIndex
from .utils.indexes import Colors


class GoalConditionedDiscreteGridWorld(DiscreteGridWorld):
    def __init__(self, map_name: str = MapsIndex.EMPTY.value):
        super().__init__(map_name=map_name)
        self.goal_coordinates = None
        self.goal = None
        self.reset_goal()

    def reset_goal(self):
        """
        Choose a goal for the agent.
        """
        self.goal_coordinates = np.flip(random.choice(np.argwhere(self.maze_map != 1)))
        self.goal = self.get_state(*self.goal_coordinates)

    def goal_reached(self):
        """
        Return a boolean True if the agent state is on the goal (and exactly on the goal since our state space is
        discrete here in reality), and false otherwise.
        """
        return (self.agent_coordinates == self.goal_coordinates).all()

    def step(self, action):
        new_x, new_y = self.get_new_coordinates(action)
        if self.is_available(new_x, new_y):
            self.agent_coordinates = new_x, new_y
            done = self.goal_reached()
            reward = -1 if not done else 0
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), reward, done
        else:
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), -1, False

    def reset(self) -> tuple:
        """
        Return the initial state, and the selected goal.
        """
        self.reset_goal()
        return super().reset(), self.goal

    def render(self, mode='human', ignore_rewards=True):
        """
        Render the whole-grid human view (get view from super class then add the goal over the image)
        """
        img = super().render(mode=mode, ignore_rewards=ignore_rewards)
        goal_x, goal_y = self.goal_coordinates
        self.place_point(img, self.get_state(goal_x, goal_y), Colors.GOAL.value)
        return img
