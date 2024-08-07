from dsg_rgl_ant.simple_rl.tasks.point_maze.environments.maze_env import MazeEnv
from dsg_rgl_ant.simple_rl.tasks.point_maze.environments.swimmer import SwimmerEnv


class SwimmerMazeEnv(MazeEnv):
    MODEL_CLASS = SwimmerEnv
