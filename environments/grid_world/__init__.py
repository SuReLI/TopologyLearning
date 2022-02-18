from environments.grid_world.discrete_grid_world import DiscreteGridWorld
from environments.grid_world.goal_conditioned_discrete_grid_world import GoalConditionedDiscreteGridWorld
from environments.utils import register_environment

register_environment(
    env_id='discrete_grid_world-v0',
    env_entry_point='environments.grid_world.discrete_grid_world:DiscreteGridWorld'
)

register_environment(
    env_id='goal_conditioned_discrete_grid_world-v0',
    env_entry_point='environments.grid_world.goal_conditioned_discrete_grid_world:GoalConditionedDiscreteGridWorld'
)
