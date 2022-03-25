import torch as T
from settings.environments_index import *

from settings.environments_index import EnvironmentsIndex
from utils.sys_fun import get_output_directory

########################################################################################################################
#                                                      SETTINGS                                                        #
########################################################################################################################

#################
#    General    #
#################

device = T.device('cuda' if T.cuda.is_available() else 'cpu')  # Device for the tensors to run on
verbose = True
global_output_directory = get_output_directory()

#################
#  Simulations  #
#################

redirect_std_output = True
nb_seeds = 5
nb_evaluations_max = 30
nb_time_steps_max_per_episodes = 50
environment_index = EnvironmentsIndex.GRID_WORLD_DISCRETE  # Environment to test the agents on
environments_rollout = False  # Rollout environment every episode if the environment present the ability to do so
pre_train_low_level_agent = True

print_reward_after_episodes = False

#################
#     Tests     #
#################

# Test video recording settings are in section "Rendering"
nb_interactions_before_evaluation = 1000
nb_tests = 20

#################
#     Plots     #
#################

colors = [  # Colors used to plot lines on topology (each simulation have its own color)
        "#c21e56",
        "#ff8243",
        "#fbec5d",
        "#00cc99",
        "#318ce7",
        "#8806ce",
        "#d473d4"
    ]

show_rewards_per_episodes = False
nb_episode_before_graph_update = 1  # None = no plot
std_area_transparency = 0.2

plot_main_side = True
plot_main_side_shape = (2, 3)
plot_agent_side = False

if not plot_agent_side and not plot_main_side:
    nb_episode_before_graph_update = None

#  Topology plot
nodes_alpha = 0.8
edges_alpha = 0.8
failed_edge_color = "#000000"
labels_color = "#000000"

#################
#   Rendering   #
#################

interactive = True

# Video settings
show_video_during_training = True
rendering_start_at_episode = 0
nb_episodes_between_two_records = None

show_video_during_tests = True
rendering_start_at_test = 0
nb_tests_between_two_records = 3

video_output_fps = 30

# Graph plot rendering settings
input_point_width = 15
input_point_color = "#3adb00"

########################################
#   Goal Reaching main file settings   #
########################################
nb_final_demo = 1

########################################################################################################################
#                                      VARIABLES DIRECTLY COMPUTED FROM SETTINGS                                       #
########################################################################################################################
