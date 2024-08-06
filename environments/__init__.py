from .environments_index import EnvironmentIndex
from .maps.maps_index import MapsIndex
from .grid_world import *
from .point_env import *
from .plot_graph_on_environment import plot_graph_on_environment
from .doom_visual_navigation import *

try:
    from .ant_maze import *
except ModuleNotFoundError:
    print("  Warning: Mujoco-py is not installed. Install it if you want to use ant-maze.")
