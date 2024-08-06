# rgl_ptim for "Reachability Graph Learning (RGL) with Pre-Training In Maze (PTIM) (instead of pre-train area)

import copy
import sys
import warnings
from datetime import datetime
import socket
import argparse
from settings import Settings
from environments import MapsIndex, EnvironmentIndex
from agents import AgentsIndex
from main import simulation
from utils import send_discord_message, EmptyWebhookLinkError, Stopwatch


# Script settings
nb_seeds = 1
if nb_seeds == 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_id", help="Simulation ID used to choose or find the output directory.", default=None)
    sim_id = parser.parse_args().sim_id
else:
    sim_id = None

webhook_link = "https://discordapp.com/api/webhooks/1040647428333895821/MOrQET25fMfR3l1V9bUcjp-JTnr8HVHDbl5TEpbSirVhYqfdTHg0LayiLo-F5faDdgDK"
if webhook_link == "":
    print("\n   You can receive a message on discord when this script is done by modifying the desired webhook link in "
          "your script at line 15.\n")

# Set up a general settings that will be the same for every simulation launched by this script
launch_name = __file__.split("/")[-1].split(".")[0]  # launch_name = File name.

# Set up a list of specific settings that we want to test (with different hyper parameters for example)
settings_set = []

settings = Settings(environment_tag=EnvironmentIndex.POINT_MAZE, agent_tag=AgentsIndex.RGL,
                    map_tag=MapsIndex.FOUR_ROOMS, pre_train_in_playground=False, simulation_id=sim_id)
settings.simulation_name = "RGL 4Rooms PM"
settings_set.append(settings)

stopwatch = Stopwatch()
stopwatch.start()
# For each specific settings in our settings list, launch <nb_seeds> simulations with these settings
for specific_settings in settings_set:
    nb_simulations = 0
    for seed_id in range(nb_seeds):

        # Find forced seed
        try:
            option_index = sys.argv.index("--seed_id")
            forced_based_seed_id = int(sys.argv[option_index + 1])
            # NB: forced_based_seed_id is the id of the first seed
            specific_settings.simulation_id = forced_based_seed_id + seed_id
        except ValueError as e:
            pass  # Option has not been specified

        print("LAUNCHING NEW SIMULATION")
        print("  - seed_id: ", seed_id)
        print("  - settings: \n", specific_settings)
        simulation(specific_settings)
        print("\n" * 3)
        nb_simulations += 1

    stopwatch.stop()
    if webhook_link != "":
        base_message = "Launch script " + specific_settings.simulation_name + " on " + socket.gethostname() + ","
        message =  base_message + " finished with a total of " + str(nb_simulations) + " simulation in " \
            + str(stopwatch.get_duration()) + " seconds."
        try:
            send_discord_message(message, webhook_link=webhook_link)
        except EmptyWebhookLinkError:
            warnings.warn("Message hasn't been send, due to an empty webhook link.")
        except ValueError:
            warnings.warn("Message hasn't been send, probably due to an invalid webhook link.")
    stopwatch.reset(start=True)
