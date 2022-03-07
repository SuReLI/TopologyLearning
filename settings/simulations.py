from math import sqrt

import gym
from gym import logger
from gym.spaces import Dict

from settings import settings
from agents import *
from settings.environments_index import EnvironmentsIndex
from utils.data_holder import DataHolder
import agents.graph_building_strategies as gbs
from environments import *

"""
This file allow us to build custom simulations, that will be run by the main file on execution.
"""

# Build environment, necessary to build agents and set their parameters
environment = gym.make(settings.environment_index.value)


# Use this environment to pre-build our simulations on it


# Use them for our simulations
class Simulation:
    def __init__(self, agent: Agent, environment_index: EnvironmentsIndex = settings.environment_index):
        self.agent = agent
        self.environment_index = environment_index
        self.color = "#000000"
        self.data_holder = DataHolder()
        self.outputs_directory = settings.global_output_directory + self.environment_index.name + "/" + \
            agent.name + "/seed_0/"
        self.agent.set_output_dir(self.outputs_directory)

        self.start_time = None
        self.end_time = None
        self.pause_total_duration = None

    def on_seed_end(self):
        self.data_holder.on_seed_end()
        self.agent.reset()


state_space = environment.observation_space
if isinstance(environment.observation_space, Dict):
    state_space = environment.observation_space["observation"]

tile_width = (1 / environment.width)
tile_height = (1 / environment.height)
allowed_tile_diff = 1 + 0.1
tolerance_radius = (tile_width + tile_height) / 2 * allowed_tile_diff
reach_distance = sqrt((tile_width / 2) ** 2 + (tile_height / 2) ** 2)

simulations = [
    Simulation(
        DFSingleAgentTL(environment=environment, tolerance_radius=tolerance_radius, state_space=state_space,
                        action_space=environment.action_space, device=settings.device, topology_manager_class=gbs.GWR,
                        name="DF + GWR", reach_distance=reach_distance)
    ),
    Simulation(
        EFSingleAgentTL(environment=environment, tolerance_radius=tolerance_radius, state_space=state_space,
                        action_space=environment.action_space, device=settings.device, topology_manager_class=gbs.GWR,
                        name="EF + GWR", reach_distance=reach_distance)
    )
]

# Give a specific color to each simulation
if len(simulations) > len(settings.colors):
    raise Exception("Too many simulations, add more colors to settings to run it.")
else:
    for simulation_id, simulation in enumerate(simulations):
        simulation.color = settings.colors[simulation_id]
