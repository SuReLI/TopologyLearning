from enum import Enum
from datetime import timedelta


class MapsIndex(Enum):
    EMPTY_ROOM = "empty_room"
    INTERMEDIATE = "intermediate"

map_name = MapsIndex.INTERMEDIATE.value

# Pre_training
pre_train_nb_episodes = 300
pre_train_nb_time_steps_per_episode = 60

# Simulation
if map_name == MapsIndex.INTERMEDIATE.value:
    nb_evaluations_max = 100
    her_max_steps = 300
else:
    raise Exception("Unknown map name.")
nb_episodes_before_evaluation = 20
nb_interactions_before_evaluation = 1000
time_before_evaluation = timedelta(minutes=2)
nb_tests_per_evaluation = 30