from enum import Enum

# Enums that list available choices for some settings (make the selection easier by allowing auto completion)
# WARNING, If an environment is added, make sure it's imported inside simulations file.


class EnvironmentsIndex(Enum):
    """
    Available environments that can be used in our simulations
    This can be used to create environments alias, ex : CartPole = "CartPole-v0" then we don't need to type version
    every time, or to list custom environment located at implem/environments.

    The value should belong to the environment name, so it can be used like : gym.make(settings.environment.value)
    """
    CART_POLE_DISCRETE = "CartPole-v1"
    LUNAR_LANDER = "LunarLanderContinuous-v2"
    LUNAR_LANDER_DISCRETE = "LunarLander-v2"
    MOUNTAIN_CAR = "MountainCarContinuous-v0"
    BIPEDAL_WALKER = "BipedalWalker-v3"
    BULLET_ENV = "InvertedPendulumBulletEnv-v0"
    INVERTED_PENDULUM_MJC = "InvertedPendulum-v2"
    GRID_WORLD_DISCRETE = "discrete_grid_world-v0"
    GOAL_CONDITIONED_GRID_WORLD_DISCRETE = "goal_conditioned_discrete_grid_world-v0"

    # The following environments requires Mujoco
    HALF_CHEETAH = "HalfCheetah-v2"
    ANT_MAZE = "AntMaze-v1"
    FETCH_PICK_PLACE = "FetchPickAndPlace-v1"
