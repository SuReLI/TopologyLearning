from environments.grid_world import DiscreteGridWorld
import random
from environments.grid_world import settings

from environments.grid_world.utils.indexes import Colors


class GoalConditionedDiscreteGridWorld(DiscreteGridWorld):
    def __init__(self, map_id=settings.map_id):
        super().__init__(map_id=map_id)
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

    def goal_reached(self):
        """
        Return a boolean True if the agent state is on the goal (and exactly on the goal since our state space is
        discrete here in reality), and false otherwise.
        """
        return self.agent_coordinates == self.goal_coordinates

    def step(self, action):
        new_x, new_y = self.get_new_coordinates(action)
        if self.is_available(new_x, new_y):
            self.agent_coordinates = new_x, new_y
            done = self.goal_reached()
            reward = -1 if not done else 1
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), reward, done, None
        else:
            return self.get_state(self.agent_coordinates[0], self.agent_coordinates[1]), -1, False, None

    def reset(self) -> tuple:
        """
        Return the initial state, and the selected goal.
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
