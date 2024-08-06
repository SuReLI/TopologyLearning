"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent.
"""
import argparse

from maps.maps_index import MapsIndex
from xml_generator import generate_xml
from design_agent_and_env import design_agent_and_env
from options import parse_options
from run_HAC import run_HAC
from gym.spaces import Box
import numpy as np
import sys
import os


def create_dir(dir_name):
    if os.path.isdir(dir_name):
        return
    dir_parts = dir_name.split("/")
    directory_to_create = ""
    for part in dir_parts:
        directory_to_create += part + "/"
        if not os.path.isdir(directory_to_create):
            try:
                os.mkdir(directory_to_create)
            except FileNotFoundError:
                print("failed to create dir " + str(directory_to_create))
                raise Exception


# Select environment
parser = argparse.ArgumentParser()
parser.add_argument('--map_id', type=int, default=0)
parser.add_argument('--simulation_id', type=int, default=0)
args = parser.parse_args()

map_id = args.map_id
simulation_id = args.simulation_id

print("Launched hac simulation on task ", map_id, " with simulation id ", simulation_id, sep="")

reset_fixed = False
reset_coordinates = False
maze_array = None
tile_size = 1.0
reset_fixed = True
uniform_goal_sampling = True
nb_test_episode = 30

if map_id == 0:
    map_name = MapsIndex.FOUR_ROOMS.value
    min_training_interactions = 700000
    nb_interactions_before_evaluation = 12000
elif map_id == 1:
    map_name = MapsIndex.MEDIUM.value
    min_training_interactions = 900000
    nb_interactions_before_evaluation = 12000
elif map_id == 2:
    map_name = MapsIndex.HARD.value
    min_training_interactions = 1600000
    nb_interactions_before_evaluation = 20000
elif map_id == 3:
    map_name = MapsIndex.JOIN_ROOMS.value
    min_training_interactions = 1600000
    nb_interactions_before_evaluation = 20000
else:
    raise Exception("Unknown value.")
maze_array, tile_size, xml_spec_path = generate_xml(map_name)
model_name = xml_spec_path.split("/")[-1]
task_name = 'HAC_' + map_name

outputs_directory = os.path.dirname(os.path.abspath(__file__)) + "/outputs/" + task_name + "/"
# Compute output dir
create_dir(outputs_directory)
if simulation_id is None:
    simulation_id = 0  # Will be incremented for each saved simulation we find.
    for filename in os.listdir(outputs_directory):
        if filename.startswith('simulation_'):
            try:
                current_id = int(filename.replace("simulation_", ""))
            except ValueError:
                continue
            simulation_id = max(simulation_id, current_id + 1)
else:
    assert isinstance(simulation_id, int)
outputs_directory += "simulation_" + str(simulation_id) + "/"
create_dir(outputs_directory)

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()

# Instantiate the agent and Mujoco environment.
# The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
agent, env = design_agent_and_env(FLAGS, model_name, reset_fixed, uniform_goal_sampling, reset_coordinates, maze_array,
                                  tile_size, simulation_id, outputs_directory)

if reset_fixed:
    assert maze_array is not None

# Begin training
run_HAC(FLAGS, env, agent, simulation_id, outputs_directory, nb_interactions_before_evaluation, min_training_interactions, nb_test_episode)
