
from gym.envs.registration import register

from environments.grid_world.discrete_grid_world import DiscreteGridWorld
from environments.grid_world.goal_conditioned_discrete_grid_world import GoalConditionedDiscreteGridWorld


register(
        id='discrete_grid_world-v0',
        entry_point='environments.grid_world.discrete_grid_world:DiscreteGridWorld'
    )

register(
        id='goal_conditioned_discrete_grid_world-v0',
        entry_point='environments.grid_world.goal_conditioned_discrete_grid_world:GoalConditionedDiscreteGridWorld'
    )
