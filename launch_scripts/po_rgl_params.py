import copy
import warnings
from datetime import datetime
import socket

from settings import Settings
from environments import MapsIndex, EnvironmentIndex
from agents import AgentsIndex
from main import simulation
from utils import send_discord_message, EmptyWebhookLinkError, Stopwatch


# Script settings
nb_seeds = 10
webhook_link = ""
if webhook_link == "":
    print("\n   You can receive a message on discord when this script is done by modifying the desired webhook link in "
          "your script at line 15.\n")

# Set up a general settings that will be the same for every simulation launched by this script
launch_name = __file__.split("/")[-1].split(".")[0]  # launch_name = File name.

# Set up a list of specific settings that we want to test (with different hyper parameters for example)
settings_set = []

# Add specific settings to this list
# new_settings = copy.deepcopy(settings)
# new_settings.simulation_name = "TC-RGL"
# new_settings.agent_tag = AgentsIndex.TC_RGL
# settings_set.append(new_settings)

settings = Settings(environment_tag=EnvironmentIndex.GRID_WORLD, agent_tag=AgentsIndex.REO_RGL,
                    map_tag=MapsIndex.MEDIUM)
settings.simulation_name = "REO-RGL, nb_nodes=200"
settings.agents_params["nb_nodes"] = 200
settings_set.append(settings)

settings = Settings(environment_tag=EnvironmentIndex.GRID_WORLD, agent_tag=AgentsIndex.REO_RGL,
                    map_tag=MapsIndex.MEDIUM)
settings.simulation_name = "REO-RGL, nb_nodes=400"
settings.agents_params["nb_nodes"] = 400
settings_set.append(settings)

settings = Settings(environment_tag=EnvironmentIndex.GRID_WORLD, agent_tag=AgentsIndex.REO_RGL,
                    map_tag=MapsIndex.MEDIUM)
settings.simulation_name = "REO-RGL, nb_nodes=600"
settings.agents_params["nb_nodes"] = 600
settings_set.append(settings)

stopwatch = Stopwatch()
stopwatch.start()
# For each specific settings in our settings list, launch <nb_seeds> simulations with these settings
nb_simulations = 0
for specific_settings in settings_set:
    for seed_id in range(nb_seeds):
        print("LAUNCHING NEW SIMULATION")
        print("  - seed_id: ", seed_id)
        print("  - settings: \n", specific_settings)
        simulation(specific_settings)
        print("\n" * 3)
        nb_simulations += 1

stopwatch.stop()
if webhook_link != "":
    base_message = "Launch script " + launch_name + " on " + socket.gethostname() + ","
    message =  base_message + " finished with a total of " + str(nb_simulations) + " simulation in " \
        + str(stopwatch.get_duration()) + " seconds."
    try:
        send_discord_message(message, webhook_link=webhook_link)
    except EmptyWebhookLinkError:
        warnings.warn("Message hasn't been send, due to an empty webhook link.")
    except ValueError:
        warnings.warn("Message hasn't been send, probably due to an invalid webhook link.")
