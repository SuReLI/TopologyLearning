import torch
from environments.maps.maps_index import MapsIndex
from environments.environments_index import EnvironmentIndex
from agents.agents_index import AgentsIndex
from datetime import timedelta

from utils import get_dict_as_str


class Settings:

    """
    We put settings in a class, so we can easily modify them in the launch script and give the modified instance
    to the main.py script.
    More useful than just a python file when we want to compare the same algorithm with some different values for a
    specific hyperparameter. We can do a for loop on the list of desired values to test, modify the instance, and give
    it to the main() (we do it in launch_simulations.py).
    """
    def __init__(self, environment_tag=EnvironmentIndex.GRID_WORLD, map_tag=MapsIndex.FOUR_ROOMS,
                 agent_tag=AgentsIndex.RGL, simulation_id=None, pre_train_in_playground=None):
        self.device = torch.device("cpu")
        self.environment_tag = environment_tag
        self.map_tag = map_tag
        self.agent_tag = agent_tag
        self.simulation_id = simulation_id
        self.simulation_name = str(self.agent_tag.value).split(".")[-1]
        # '--> A simulation name specific to this simulation. Simulation name is important since it will be used to set
        #      the default outputs directory name. Run many simulation with the same agent but with different
        #      hyperparameter will lead to have one directory that contains every simulation, without knowing which
        #      simulation have which parameters. (NB: as a security, simulations settings is always saved in a
        #      "simulation_settings.pkl" file inside the output directory, but set a relevant name make outputs
        #      management more handy).
        self.redirect_std_output = False

        # Pre_training
        if pre_train_in_playground is None:
            self.pre_train_in_playground = agent_tag in [AgentsIndex.RGL, AgentsIndex.REO_RGL, AgentsIndex.TC_RGL]
        else:
            self.pre_train_in_playground = pre_train_in_playground

        if self.environment_tag == EnvironmentIndex.GRID_WORLD:
            self.pre_train_nb_time_steps_per_episode = 70
            if agent_tag in [AgentsIndex.RGL, AgentsIndex.REO_RGL]:
                self.pre_train_nb_episodes = 100
            elif agent_tag in [AgentsIndex.TC_RGL]:
                self.pre_train_nb_episodes = 200
            elif agent_tag in [AgentsIndex.SGM, AgentsIndex.SORB]:
                self.pre_train_nb_episodes = 300
            elif agent_tag != AgentsIndex.DQN:
                raise Exception("")
        elif self.environment_tag == EnvironmentIndex.POINT_MAZE:
            self.pre_train_nb_time_steps_per_episode = 100
            self.pre_train_nb_episodes = 70
        elif self.environment_tag == EnvironmentIndex.ANT_MAZE:
            self.pre_train_nb_time_steps_per_episode = 150
            self.pre_train_nb_episodes = 2200
        else:
            raise NotImplementedError("Unknown environment tag.")

        self.control_only_agent_max_steps = 100  # By brute agent we mean agent that do not plan sub-goals (like DQN, SAC, ...)
        if self.map_tag == MapsIndex.FOUR_ROOMS:
            self.nb_interactions_max = 1e5
            if environment_tag == EnvironmentIndex.ANT_MAZE:
                self.nb_interactions_max = 500000
        if self.map_tag == MapsIndex.MEDIUM:
            self.nb_interactions_max = 150000
            if environment_tag == EnvironmentIndex.ANT_MAZE:
                self.nb_interactions_max = 700000
        if self.map_tag == MapsIndex.JOIN_ROOMS:
            self.nb_interactions_max = 210000
            if environment_tag == EnvironmentIndex.ANT_MAZE:
                self.nb_interactions_max = 1200000
        elif self.map_tag == MapsIndex.HARD:
            self.nb_interactions_max = 250000
            if environment_tag == EnvironmentIndex.ANT_MAZE:
                self.nb_interactions_max = 1200000
        self.nb_interactions_before_evaluation = 1000
        if environment_tag == EnvironmentIndex.ANT_MAZE:
            self.nb_interactions_before_evaluation = 6000
        self.nb_episodes_before_evaluation = 10  # For agents with fixed episode duration
        self.nb_tests_per_evaluation = 30
        self.agents_params = {}

    def get_simulation_description(self):
        return str(self.environment_tag.value) + " " + str(self.map_tag.value) + ", with name " + self.simulation_name

    def __str__(self):
        return get_dict_as_str(vars(self))
