from grid_world.environment.maps.maps_index import MapsIndex
from datetime import timedelta


map_name = MapsIndex.FOUR_ROOMS.value

# Pre_training
pre_train_nb_episodes = 300
pre_train_nb_time_steps_per_episode = 60

# Simulation
if map_name == MapsIndex.FOUR_ROOMS.value:
    nb_evaluations_max = 100
    dqn_max_steps = 300
if map_name == MapsIndex.MEDIUM.value:
    nb_evaluations_max = 100
    dqn_max_steps = 300
elif map_name == MapsIndex.HARD.value:
    nb_evaluations_max = 300
    dqn_max_steps = 1000
elif map_name == MapsIndex.EXTREME.value:
    nb_evaluations_max = 700
    dqn_max_steps = 2000
nb_episodes_before_evaluation = 20
nb_interactions_before_evaluation = 1000
time_before_evaluation = timedelta(minutes=2)
nb_tests_per_evaluation = 30