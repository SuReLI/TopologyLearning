import torch as T
import numpy as np

from src.utils.sys_fun import get_output_directory

########################################################################################################################
#                                                      SETTINGS                                                        #
########################################################################################################################

#################
#    General    #
#################

device = T.device('cuda' if T.cuda.is_available() else 'cpu')  # Device for the tensors to run on
verbose = True

#################
#  Simulations  #
#################

# Pre-training
pre_train_initial_random_exploration_duration = 50
nb_pretrain_initial_random_explorations = 4

nb_seeds = 4

# Settings for HER demonstration simulations, unused by topology learning:
nb_episodes_per_simulation = 300
nb_episodes_before_tests = 20
nb_episodes_before_plots = 20
episode_length = 80  # How many interactions our agent is allowed to do to reach a goal

# Settings topology learning only:
nb_evaluations_max = 50

# Misc
print_reward_after_episodes = False
redirect_std_output = False

#################
#    Ant Maze   #
#################
ant_maze_episode_length = 200
ant_maze_pre_training_duration = 5000
ant_maze_pre_train_reachability_threshold = 0.8
ant_maze_pre_train_velocity_threshold = 0.8
ant_maze_pre_train_angle_threshold = 0.3
ant_maze_nodes_reachability_threshold = 0.4
ant_maze_reach_distance_threshold = 0.5
ant_maze_nb_interactions_before_evaluation = 2000
ant_maze_nb_evaluations_max = 30


#################
#   Point Maze  #
#################
point_maze_episode_length = 200
point_maze_pre_training_duration = 2000
point_maze_pre_train_reachability_threshold = 0.6
point_maze_pre_train_velocity_threshold = 0.8
point_maze_pre_train_angle_threshold = 0.3
point_maze_nodes_reachability_threshold = 0.4
point_maze_reach_distance_threshold = 0.5
point_maze_nb_interactions_before_evaluation = 2000
point_maze_nb_evaluations_max = 30


#################
#     Tests     #
#################

# Test video recording settings are in section "Rendering"
nb_interactions_before_evaluation = 2000
nb_tests = 40

#################
#     Plots     #
#################

plots_window_width = 1400
plots_window_height = 900
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
nb_interactions_before_graph_update = nb_interactions_before_evaluation  # None = no plot
std_area_transparency = 0.2

plot_main_side = True
plot_main_side_shape = (2, 2)
plot_agent_side = True

if not plot_agent_side and not plot_main_side:
    nb_episode_before_graph_update = None

#  Topology plot
nodes_size = 100
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
