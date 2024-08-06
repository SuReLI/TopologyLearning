"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import math
import os
import pickle
import random
import sys
from copy import deepcopy, copy
from os.path import isdir
from statistics import mean

import networkx as nx
import numpy
import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Discrete

from agents import AgentsIndex, Agent, GoalConditionedAgent, RGL, REO_RGL, TC_RGL, SORB, SGM, TILO, HER, SAC, DQN
from environments import MapsIndex, GoalConditionedDiscreteGridWorld, EnvironmentIndex, GoalConditionedPointEnv, AntMaze
from environments.ant_maze.HAC_ant_environment import HACAntEnvironment
from settings import Settings
from utils import create_dir, save_image, Stopwatch


class SimulationInformation:
    def __init__(self):
        self.id = 0
        self.output_directory = None
        self.seeds = {}
        self.outputs_directory = ""
        self.nb_interactions = 0

        # Stopwatches

        self.pre_train_nb_interactions = 0
        self.pre_training_stopwatch = Stopwatch()
        # The durations measured by the two stopwatches bellow is included in the one measured by the stopwatch above.
        self.pre_training_learning_stopwatch = Stopwatch()
        self.pre_training_env_step_stopwatch = Stopwatch()
        # '--> Cost of the environment to process. Independent of the agent. Same for the one bellow.

        self.training_stopwatch = Stopwatch()
        # The durations measured by every stopwatch bellow is included in the one measured by the stopwatch above.
        self.learning_stopwatch = Stopwatch()  # Also include the time to take an action. (aka cost specific to algorithm)
        self.env_steps_stopwatch = Stopwatch()
        self.evaluation_stopwatch = Stopwatch()
        self.output_generation_stopwatch = Stopwatch()

    def dict(self):
        return {
            "simulation id": str(self.id),
            "output_directory": str(self.output_directory),
            "seeds": self.seeds,
            "outputs directory": str(self.outputs_directory),
            "nb interactions": str(self.nb_interactions),
            "pre training nb interactions": str(self.pre_train_nb_interactions),
            "pre training duration": str(self.pre_training_stopwatch),
            "pre training learning duration": str(self.pre_training_learning_stopwatch),
            "pre training env step duration": str(self.pre_training_env_step_stopwatch),
            "training duration": str(self.training_stopwatch),
            "learning duration": str(self.learning_stopwatch),
            "env_steps duration": str(self.env_steps_stopwatch),
            "evaluation duration": str(self.evaluation_stopwatch),
            "output generation duration": str(self.output_generation_stopwatch),
        }

def simulation(settings: Settings):
    """
    Build a simulation, run it, and save its results in a new output directory.
    @param settings: simulation settings.
    """
    assert isinstance(settings, Settings)

    agent, environment, simulation_information = init(settings)
    agent.reset()
    print("#################")
    print("Running a simulation on " + settings.environment_tag.name + " ...")
    print("Agent ", agent.name, sep='')
    print("Map ", settings.map_tag.value, sep='')
    print("Simulation id: ", settings.map_tag.value, "/", agent.name, "/simulation_", simulation_information.id, sep='')
    print("#################")
    if not os.path.isdir(simulation_information.outputs_directory + "save/"):
        # There is no save for the current simulation, so the pre-training should be done.

        if isinstance(agent, RGL) or isinstance(agent, REO_RGL) or isinstance(agent, SORB):
            if settings.pre_train_in_playground:
                pre_train_env_map_name = str(MapsIndex.EMPTY.value)
            else:
                pre_train_env_map_name = str(settings.map_tag.value)
            if settings.environment_tag == EnvironmentIndex.GRID_WORLD:
                pre_train_environment = GoalConditionedDiscreteGridWorld(map_name=pre_train_env_map_name)
            elif settings.environment_tag == EnvironmentIndex.POINT_MAZE:
                pre_train_environment = GoalConditionedPointEnv(map_name=pre_train_env_map_name)
            elif settings.environment_tag == EnvironmentIndex.ANT_MAZE:
                pre_train_environment = AntMaze(maze_name=pre_train_env_map_name, random_orientation=True)
            else:
                raise NotImplementedError("Unknown environment type")

            # Load pre_training information
            start_state, reached_goals = pre_train_gc_agent(settings, simulation_information, pre_train_environment, agent)

            if isinstance(agent, REO_RGL):
                oracle = environment.get_oracle()
                filtered_oracle = []
                for state in oracle:
                    if environment.is_available(*environment.get_coordinates(state)):
                        filtered_oracle.append(state)
                agent.on_pre_training_done(start_state, reached_goals, filtered_oracle)
            elif isinstance(agent, SORB):  # NB: SGM is a subclass of SORB
                agent.on_pre_training_done()
            else:
                agent.on_pre_training_done(start_state, reached_goals)

    run_simulation(settings, simulation_information, agent, environment)
    if hasattr(agent, "reachability_graph"):
        print("nb_nodes = ", len(list(agent.reachability_graph.nodes)))


def init(settings: Settings):
    simulation_information = SimulationInformation()

    # Init seeds (they will be stored in an output file in the output directory after the simulation)
    seed = random.randrange(sys.maxsize)
    simulation_information.seeds.update(torch=copy(seed))
    torch.manual_seed(seed)

    seed = random.randrange(2 ** 32 - 1)  # Maximum seed range allowed by numpy
    simulation_information.seeds.update(numpy=copy(seed))
    numpy.random.seed(seed)

    seed = random.randrange(sys.maxsize)
    simulation_information.seeds.update(random=copy(seed))
    random.seed(seed)

    # Initialise environment
    if settings.environment_tag == EnvironmentIndex.GRID_WORLD:
        environment = GoalConditionedDiscreteGridWorld(map_name=settings.map_tag.value)
    elif settings.environment_tag == EnvironmentIndex.POINT_MAZE:
        environment = GoalConditionedPointEnv(map_name=settings.map_tag.value)
    elif settings.environment_tag == EnvironmentIndex.ANT_MAZE:
        # environment = AntMaze(maze_name=settings.map_tag.value, show=True)  # TODO: ENV
        environment = AntMaze(maze_name=settings.map_tag.value)
    else:
        raise NotImplementedError("Unknown environment type")

    # Initialise agent
    if settings.agent_tag == AgentsIndex.RGL:
        if settings.environment_tag == EnvironmentIndex.GRID_WORLD:
            settings.agents_params["exploration_goal_range"] = 5
            if settings.pre_train_in_playground:
                settings.agents_params["edges_distance_threshold"] = \
                    settings.agents_params.get("edges_distance_threshold", 0.2)
                settings.agents_params["nodes_distance_threshold"] = \
                    settings.agents_params.get("nodes_distance_threshold", 0.1)
            else:
                if settings.map_tag == MapsIndex.FOUR_ROOMS:
                    settings.agents_params["edges_distance_threshold"] = \
                        settings.agents_params.get("edges_distance_threshold", 0.2)
                    settings.agents_params["nodes_distance_threshold"] = \
                        settings.agents_params.get("nodes_distance_threshold", 0.1)
                elif settings.map_tag == MapsIndex.MEDIUM:
                    settings.agents_params["edges_distance_threshold"] = \
                        settings.agents_params.get("edges_distance_threshold", 0.4)
                    settings.agents_params["nodes_distance_threshold"] = \
                        settings.agents_params.get("nodes_distance_threshold", 0.2)
                elif settings.map_tag == MapsIndex.HARD:
                    settings.agents_params["edges_distance_threshold"] = \
                        settings.agents_params.get("edges_distance_threshold", 0.4)
                    settings.agents_params["nodes_distance_threshold"] = \
                        settings.agents_params.get("nodes_distance_threshold", 0.2)
                elif settings.map_tag == MapsIndex.JOIN_ROOMS:
                    settings.agents_params["edges_distance_threshold"] = \
                        settings.agents_params.get("edges_distance_threshold", 0.3)
                    settings.agents_params["nodes_distance_threshold"] = \
                        settings.agents_params.get("nodes_distance_threshold", 0.15)
            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 1)
            control_algorithm = DQN
            agent = RGL(TILO, control_algorithm, environment.state_space, environment.action_space,
                        default_state=environment.reset()[0], **settings.agents_params)
        elif settings.environment_tag == EnvironmentIndex.POINT_MAZE:
            settings.agents_params["exploration_goal_range"] = 4
            settings.pre_train_nb_episodes = 150
            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 0.8)

            settings.agents_params["edges_distance_threshold"] = \
                settings.agents_params.get("edges_distance_threshold", 0.08)
            settings.agents_params["nodes_distance_threshold"] = \
                settings.agents_params.get("nodes_distance_threshold", 0.04)

            control_algorithm = SAC
            agent = RGL(TILO, control_algorithm, environment.state_space, environment.action_space,
                        default_state=environment.reset()[0], **settings.agents_params)
        elif settings.environment_tag == EnvironmentIndex.ANT_MAZE:

            settings.pre_train_nb_episodes = 3000
            settings.pre_train_nb_time_steps_per_episode = 150
            settings.agents_params["edges_distance_threshold"] = \
                settings.agents_params.get("edges_distance_threshold", 0.3)
            settings.agents_params["nodes_distance_threshold"] = \
                settings.agents_params.get("nodes_distance_threshold", 0.1)

            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 0.7)
            settings.agents_params["exploration_goal_range"] = \
                settings.agents_params.get("exploration_goal_range", 6)
            settings.agents_params["exploration_duration"] = \
                settings.agents_params.get("exploration_duration", 150)
            settings.agents_params["max_steps_to_reach"] = \
                settings.agents_params.get("max_steps_to_reach", 150)
            state_to_goal_filter = [True] * 2 + [False] * 27
            goal_space = Box(low=environment.state_space.low[state_to_goal_filter],
                             high=environment.state_space.high[state_to_goal_filter])
            control_algorithm = TILO(SAC, state_space=environment.state_space, action_space=environment.action_space,
                         goal_space=goal_space, batch_size=500, buffer_max_size=1e6, actor_alpha=0.1)

            agent = RGL(TILO, control_algorithm, environment.state_space, environment.action_space,
                        goal_space=goal_space, default_state=environment.reset()[0], **settings.agents_params)

        else:
            settings.agents_params["edges_distance_threshold"] = \
                settings.agents_params.get("edges_distance_threshold", 0.1)
            settings.agents_params["nodes_distance_threshold"] = \
                settings.agents_params.get("nodes_distance_threshold", 0.017)
            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 0.8)
            raise Exception("Unknown environment type.")
    elif settings.agent_tag == AgentsIndex.TC_RGL:
        if settings.environment_tag == EnvironmentIndex.GRID_WORLD:
            settings.agents_params["edges_distance_threshold"] = \
                settings.agents_params.get("edges_distance_threshold", 0.4)
            settings.agents_params["nodes_distance_threshold"] = \
                settings.agents_params.get("nodes_distance_threshold", 0.2)
            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 1)
        elif settings.environment_tag == EnvironmentIndex.POINT_MAZE:
            settings.agents_params["exploration_goal_range"] = 4
            settings.pre_train_nb_episodes = 600
            settings.agents_params["edges_distance_threshold"] = \
                settings.agents_params.get("edges_distance_threshold", 0.1)
            settings.agents_params["nodes_distance_threshold"] = \
                settings.agents_params.get("nodes_distance_threshold", 0.03)
            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 0.8)
        else:
            settings.agents_params["edges_distance_threshold"] = \
                settings.agents_params.get("edges_distance_threshold", 0.2)
            settings.agents_params["nodes_distance_threshold"] = \
                settings.agents_params.get("nodes_distance_threshold", 0.107)
            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 0.8)

        if settings.environment_tag == EnvironmentIndex.GRID_WORLD:
            control_algorithm = DQN
            agent = TC_RGL(TILO, control_algorithm, environment.state_space, environment.action_space,
                        default_state=environment.reset()[0], **settings.agents_params)
        elif settings.environment_tag == EnvironmentIndex.POINT_MAZE:
            control_algorithm = SAC
            agent = TC_RGL(TILO, control_algorithm, environment.state_space, environment.action_space,
                        default_state=environment.reset()[0], **settings.agents_params)
    elif settings.agent_tag == AgentsIndex.REO_RGL:
        control_algorithm = DQN if isinstance(environment.action_space, Discrete) else SAC
        if settings.environment_tag == EnvironmentIndex.GRID_WORLD:
            settings.agents_params["edges_distance_threshold"] = \
                settings.agents_params.get("edges_distance_threshold", 0.2)
            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 1)
        elif settings.environment_tag == EnvironmentIndex.POINT_MAZE:
            settings.agents_params["edges_distance_threshold"] = \
                settings.agents_params.get("edges_distance_threshold", 0.03)
            settings.agents_params["tolerance_radius"] = \
                settings.agents_params.get("tolerance_radius", 0.8)

        nb_nodes = 150
        if settings.map_tag == MapsIndex.FOUR_ROOMS:
            nb_nodes = 400
        elif settings.map_tag == MapsIndex.MEDIUM:
            nb_nodes = 500
        elif settings.map_tag == MapsIndex.HARD:
            nb_nodes = 700
        elif settings.map_tag == MapsIndex.JOIN_ROOMS:
            nb_nodes = 900
        agent = REO_RGL(TILO, control_algorithm, environment.state_space, environment.action_space,
                        nb_nodes=nb_nodes, default_state=environment.reset()[0], **settings.agents_params)
    elif settings.agent_tag == AgentsIndex.DQN:
        agent = HER(DQN, environment.state_space, environment.action_space)
    elif settings.agent_tag == AgentsIndex.SAC:
        agent = HER(SAC, environment.state_space, environment.action_space)
    elif settings.agent_tag == AgentsIndex.SGM:

        if settings.environment_tag == EnvironmentIndex.GRID_WORLD:
            settings.pre_train_nb_episodes = 500
            settings.agents_params["nb_nodes"] = settings.agents_params.get("nb_nodes", 1000)
            settings.agents_params["reachability_threshold"] = 0.1
            if settings.map_tag == MapsIndex.FOUR_ROOMS:
                settings.agents_params["node_pruning_threshold"] = 2
                settings.agents_params["max_edges_length"] = 5
            if settings.map_tag == MapsIndex.MEDIUM:
                settings.agents_params["node_pruning_threshold"] = 3
                settings.agents_params["max_edges_length"] = 6
            if settings.map_tag == MapsIndex.HARD:
                settings.agents_params["node_pruning_threshold"] = 3
                settings.agents_params["max_edges_length"] = 6
            if settings.map_tag == MapsIndex.JOIN_ROOMS:
                settings.agents_params["node_pruning_threshold"] = 2
                settings.agents_params["max_edges_length"] = 5
            if settings.map_tag == MapsIndex.FOUR_ROOMS:
                settings.agents_params["nb_nodes"] = 1400
            if settings.map_tag == MapsIndex.MEDIUM:
                settings.agents_params["nb_nodes"] = 1400
            if settings.map_tag == MapsIndex.JOIN_ROOMS:
                settings.agents_params["nb_nodes"] = 1600
            if settings.map_tag == MapsIndex.HARD:
                settings.agents_params["nb_nodes"] = 1800

        elif settings.environment_tag == EnvironmentIndex.POINT_MAZE:
            settings.pre_train_nb_time_steps_per_episode = 20
            settings.agents_params["node_pruning_threshold"] = 3
            settings.agents_params["max_edges_length"] = 7

            settings.agents_params["nb_nodes"] = 1000
            if settings.map_tag == MapsIndex.FOUR_ROOMS:
                settings.pre_train_nb_episodes = 1500
                settings.agents_params["nb_nodes"] = 1400
            if settings.map_tag == MapsIndex.MEDIUM:
                settings.pre_train_nb_episodes = 2000
                settings.agents_params["nb_nodes"] = 1400
            if settings.map_tag == MapsIndex.JOIN_ROOMS:
                settings.pre_train_nb_episodes = 2500
                settings.agents_params["nb_nodes"] = 1600
            if settings.map_tag == MapsIndex.HARD:
                settings.pre_train_nb_episodes = 2700
                settings.agents_params["nb_nodes"] = 1800

        agent = SGM(agent_wrapper=HER, state_space=environment.state_space, action_space=environment.action_space,

                    **settings.agents_params)
    else:
        raise Exception("Unknown agent name " + settings.agent_tag.value)

    # Create a directory for this simulation and set the stdout into a file inside this directory
    simulation_information.outputs_directory = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir("/scratch/disc/h.bonnavaud"):  # Test if I'm on pando or not
        split_path = simulation_information.outputs_directory.split("/")
        split_path[1] = "scratch"
        simulation_information.outputs_directory = "/".join(split_path)
        print("output path = ", split_path)
    simulation_information.outputs_directory += "/outputs/simulations/" + \
        settings.environment_tag.value + "/" + settings.map_tag.value + "/" + agent.name + "/"
    create_dir(simulation_information.outputs_directory)

    if settings.simulation_id is None:
        # Get filename: Iterates through saved simulations to find an available id
        simulation_id = 0  # Will be incremented for each saved simulation we find.
        for filename in os.listdir(simulation_information.outputs_directory):
            if filename.startswith('simulation_'):
                try:
                    current_id = int(filename.replace("simulation_", ""))
                except ValueError:
                    continue
                simulation_id = max(simulation_id, current_id + 1)
        simulation_information.simulation_id = simulation_id
    else:
        simulation_information.simulation_id = settings.simulation_id


    simulation_information.outputs_directory += "simulation_" + str(simulation_information.simulation_id) + "/"

    if not os.path.isdir(simulation_information.outputs_directory + "save/"):  # TODO remove
        create_dir(simulation_information.outputs_directory)

    # Redirect stdout to a file within this directory
    if settings.redirect_std_output:
        sys.stdout = open(simulation_information.outputs_directory + 'standard_output.txt', 'w')

    return agent, environment, simulation_information


def pre_train_gc_agent(settings: Settings, simulation_information: SimulationInformation, environment, agent):
    simulation_information.pre_training_stopwatch.start()
    print("Pretraining low level agent ... please wait a bit ...")
    pre_training_agent = agent.control_policy
    nb_interactions = 0
    reached_goals = []
    results = []
    start_state = None
    if isinstance(agent, TC_RGL):
        last_trajectory = []
    simulation_information.pre_training_learning_stopwatch.start()

    if isinstance(environment, AntMaze) and isinstance(agent, RGL):
        nb_episodes_before_reset = 5
        nb_episodes_since_last_reset = 0
        max_goal_distance = 0
    for episode_id in range(settings.pre_train_nb_episodes):
        simulation_information.pre_training_learning_stopwatch.stop()
        simulation_information.pre_training_env_step_stopwatch.start()
        if isinstance(environment, AntMaze) and isinstance(agent, RGL):
            reset_state, goal = environment.reset()
        else:
            state, goal = environment.reset()
        simulation_information.pre_training_env_step_stopwatch.stop()
        simulation_information.pre_training_learning_stopwatch.start()

        if isinstance(environment, AntMaze) and isinstance(agent, RGL):

            # Ant-Maze pretraining reset procedure define in the paper
            if nb_episodes_since_last_reset == nb_episodes_before_reset:
                nb_episodes_since_last_reset = 0
            if nb_episodes_since_last_reset == 0:
                state = reset_state
            else:  # Don't reset
                state[:2] = reset_state[:2]
            nb_episodes_since_last_reset += 1
            # SAMPLE GOAL
            # Sample radius
            r = random.random() * 2 * math.pi
            # Sample distance
            d = math.sqrt(random.random()) * max_goal_distance
            # Compute goal coordinates
            goal[:2] = np.array([math.cos(r) * d, math.sin(r) * d]) + state[:2].copy()
            environment.goal = goal

        if isinstance(agent, TC_RGL):
            last_trajectory.append(state)
        # Ate the end of the pre-training, the agent will have an accuracy of 1 (100 %). Then, we add every goals to
        # the reached goals list, so we can have more goals, even if they hasn't been reached yet.
        reached_goals.append(goal)
        if start_state is None:
            start_state = state.copy()
        pre_training_agent.start_episode(state, goal)

        reached = False
        done = False

        for interaction_id in range(settings.pre_train_nb_time_steps_per_episode):

            action = pre_training_agent.action(state)

            simulation_information.pre_training_learning_stopwatch.stop()
            simulation_information.pre_training_env_step_stopwatch.start()
            state, reward, reached = environment.step(action)
            simulation_information.pre_training_env_step_stopwatch.stop()
            simulation_information.pre_training_learning_stopwatch.start()

            if isinstance(agent, TC_RGL):
                last_trajectory.append(state)

            pre_training_agent.process_interaction(action, reward, state, done, learn=True)

            nb_interactions += 1
            if reached:
                break

        results.append(1 if reached else 0)

        if isinstance(environment, (AntMaze, HACAntEnvironment)) and isinstance(agent, RGL):
            if reached:
                if isinstance(environment, AntMaze):
                    max_goal_distance = min(max_goal_distance + 0.1, 7)
                else:
                    max_goal_distance = min(max_goal_distance + 0.1, 5)
            else:
                max_goal_distance = max(max_goal_distance - 0.1, 0)

        if len(results) > 20:
            last_20_average = mean(results[-20:])
        else:
            last_20_average = mean(results)
        if isinstance(environment, AntMaze) and isinstance(agent, RGL):
            print("Episode ", episode_id, "; average accuracy over last 20 episodes = ", last_20_average * 100,
                  "%; max_goal_distance = ", max_goal_distance, "        ", sep="", end="\r")
        else:
            print("Episode ", episode_id, "; average accuracy over last 20 episodes = ", last_20_average * 100, "%        ",
                  end="\r")

        if episode_id == settings.pre_train_nb_episodes - 2:
            print(end="\x1b[2K")
        pre_training_agent.stop_episode()

        if isinstance(agent, TC_RGL):
            agent.store_tc_training_samples(last_trajectory)
            last_trajectory = []
    simulation_information.pre_training_learning_stopwatch.stop()
    simulation_information.pre_training_stopwatch.stop()
    simulation_information.pre_train_nb_interactions = nb_interactions
    simulation_information.nb_interactions += nb_interactions
    return start_state, reached_goals


def run_simulation(settings: Settings, simulation_information: SimulationInformation,
                   agent: Agent, environment):

    evaluations_results = []
    evaluations_abscissa = []  # at which time step does each evaluation has been taken (accuracy graph abscissa)
    nodes_in_graph = []
    pruned_nodes_in_graph = []
    edges_in_graph = []

    # Train
    nb_interactions = 0
    nb_evaluations = 0
    episode_id = 0

    if os.path.isdir(simulation_information.outputs_directory + "save/"):  # TODO remove
        # Load the simulation inside this directory

        # Save simulation
        directory = simulation_information.outputs_directory + "save/"

        with open(directory + "simulation_variables.pkl", "rb") as f:
            simulation_variables = pickle.load(f)
        with open(directory + "simulation_information.pkl", "rb") as f:
            simulation_information = pickle.load(f)
        with open(directory + "settings.pkl", "rb") as f:
            settings = pickle.load(f)
        agent.load(directory + "agent/")

        evaluations_results = simulation_variables["evaluations_results"]
        evaluations_abscissa = simulation_variables["evaluations_abscissa"]
        nodes_in_graph = simulation_variables["nodes_in_graph"]
        pruned_nodes_in_graph = simulation_variables["pruned_nodes_in_graph"]
        edges_in_graph = simulation_variables["edges_in_graph"]
        nb_interactions = simulation_variables["nb_interactions"]
        nb_evaluations = simulation_variables["nb_evaluations"]
        episode_id = simulation_variables["episode_id"]

    simulation_information.training_stopwatch.start()

    while True:
        simulation_information.env_steps_stopwatch.start()
        state, goal = environment.reset()
        simulation_information.env_steps_stopwatch.stop()
        advancement = nb_interactions / settings.nb_interactions_max * 100
        print("Simulation ", simulation_information.id, ", episode ", episode_id, ", advancement: ",
              advancement, "%" +
              (";  last_eval_grade = " + str(evaluations_results[-1])) if evaluations_results else "", sep='', end="\r")

        simulation_information.learning_stopwatch.start()

        if isinstance(agent, RGL):
            agent.start_episode(state)
        else:
            agent.start_episode(state, goal)

        simulation_information.learning_stopwatch.stop()

        local_interaction_id = 0
        pruned_edges = []

        trajectory = [state.copy()]
        while not episode_done(settings, agent, local_interaction_id):
            # Evaluation if needed
            if nb_interactions >= (nb_evaluations *  settings.nb_interactions_before_evaluation):
                nb_evaluations += 1
                result, goals, results = evaluation(settings, simulation_information, agent, nb_evaluations)
                evaluations_abscissa.append(simulation_information.pre_train_nb_interactions + nb_interactions)
                evaluations_abscissa.append(simulation_information.pre_train_nb_interactions + nb_interactions)
                evaluations_results.append(result)

                # Save training information
                directory = simulation_information.outputs_directory + "/graph_images/"
                create_dir(directory)
                if isinstance(agent, RGL) or isinstance(agent, REO_RGL) or isinstance(agent, SORB):
                    generate_graph_image(environment, agent.reachability_graph, directory,
                                         "evaluation_" + str(nb_evaluations) + ".png")

                if isinstance(agent, RGL) or isinstance(agent, REO_RGL):
                    nodes_in_graph.append(len(agent.reachability_graph.nodes))
                    agent_edges_costs = [data["cost"] for _, _, data in agent.reachability_graph.edges(data=True)]
                    pruned_nodes_in_graph.append(np.where(np.array(agent_edges_costs) == float("inf"))[0].shape[0])
                    edges_in_graph.append(len(agent.reachability_graph.edges))
            simulation_information.learning_stopwatch.start()
            action = agent.action(state)
            simulation_information.learning_stopwatch.stop()

            simulation_information.env_steps_stopwatch.start()
            state, reward, done = environment.step(action)
            simulation_information.env_steps_stopwatch.stop()
            trajectory.append(state.copy())

            local_interaction_id += 1
            nb_interactions += 1
            simulation_information.learning_stopwatch.start()
            pruned_edge = agent.process_interaction(action, reward, state, done)
            if pruned_edge is not None:
                pruned_edges.append(pruned_edge)

            simulation_information.learning_stopwatch.stop()

        # Save an image that represent the episode
        directory = simulation_information.outputs_directory + "/exploration_image/"
        create_dir(directory)
        if isinstance(agent, RGL) or isinstance(agent, TC_RGL):
            if agent.under_exploration:
                generate_graph_image(environment, agent.reachability_graph, directory, str(episode_id) + ".png",
                                     exploration_trajectory=agent.last_exploration_trajectory,
                                     sampled_goal=agent.sampled_exploration_goal,
                                     exploration_node=agent.get_node_attribute(agent.exploration_node, "state"))
            else:

                generate_graph_image(environment, agent.reachability_graph, directory, str(episode_id) + ".png",
                                     exploration_goal=agent.next_way_point,
                                     sampled_goal=agent.sampled_exploration_goal)

        image = environment.render(ignore_goal=True, ignore_rewards=True)

        episode_id += 1

        agent.stop_episode()
        if nb_interactions > settings.nb_interactions_max:
            break

        if episode_id != 0 and episode_id % 10 == 0:  # TODO: remove
            # Pause stopwatches
            simulation_information.training_stopwatch.stop()

            # Save simulation
            directory = simulation_information.outputs_directory + "save/"
            create_dir(directory)
            simulation_variables = {
                "evaluations_results": evaluations_results,
                "evaluations_abscissa": evaluations_abscissa,
                "nodes_in_graph": nodes_in_graph,
                "pruned_nodes_in_graph": pruned_nodes_in_graph,
                "edges_in_graph": edges_in_graph,
                "nb_interactions": nb_interactions,
                "nb_evaluations": nb_evaluations,
                "episode_id": episode_id
            }

            with open(directory + "simulation_variables.pkl", "wb") as f:
                pickle.dump(simulation_variables, f)
            with open(directory + "simulation_information.pkl", "wb") as f:
                pickle.dump(simulation_information, f)
            with open(directory + "settings.pkl", "wb") as f:
                pickle.dump(settings, f)
            agent.save(directory + "agent/")

            # Play stopwatches
            simulation_information.training_stopwatch.start()

    simulation_information.output_generation_stopwatch.start()
    if isinstance(agent, RGL) or isinstance(agent, REO_RGL):
        simulation_information.nodes_in_final_graph = len(agent.reachability_graph.nodes())
        simulation_information.edges_in_final_graph = len(agent.reachability_graph.edges())

    print(end="\x1b[2K")
    print("Simulation ", simulation_information.id, " advancement: Done.", sep='')
    print("accuracy_evolution = ", evaluations_results, sep='')

    # Save simulation results
    output_directory = simulation_information.outputs_directory
    with open(output_directory + 'simulation_settings.pkl', 'wb') as f:
        pickle.dump(str(settings), f)
    with open(output_directory + 'simulation_abscissa.pkl', 'wb') as f:
        pickle.dump([(evaluation_id + 1) * settings.nb_interactions_before_evaluation + simulation_information.nb_interactions
                     for evaluation_id in range(len(evaluations_results))], f)
    simulation_information.nb_interactions += nb_interactions
    with open(output_directory + 'simulation_results.pkl', 'wb') as f:
        pickle.dump(evaluations_results, f)

    if isinstance(agent, RGL) or isinstance(agent, REO_RGL):
        # Save graph information
        with open(output_directory + 'simulation_nodes_in_graph.pkl', 'wb') as f:
            pickle.dump(nodes_in_graph, f)
        with open(output_directory + 'simulation_pruned_nodes_in_graph.pkl', 'wb') as f:
            pickle.dump(pruned_nodes_in_graph, f)
        with open(output_directory + 'simulation_edges_in_graph.pkl', 'wb') as f:
            pickle.dump(edges_in_graph, f)
        with open(output_directory + "agent_final_graph.pkl", 'wb') as f:
            pickle.dump(agent.reachability_graph, f)

        # Save graph representation if exists
        generate_graph_image(environment, agent.reachability_graph, output_directory, "final_graph_representation.png")

    simulation_information.output_generation_stopwatch.stop()
    simulation_information.training_stopwatch.stop()
    with open(output_directory + 'simulation_info.pkl', 'wb') as f:
        pickle.dump(simulation_information.dict(), f)

    if settings.redirect_std_output:
        sys.stdout = sys.__stdout__
    print(" >> simulation saved in directory ", output_directory)


def evaluation(settings: Settings, simulation_information, agent, nb_evaluations):
    # Get an agent copy and prepare it to the test
    simulation_information.evaluation_stopwatch.start()
    test_agent = deepcopy(agent)

    if settings.environment_tag == EnvironmentIndex.GRID_WORLD:
        environment = GoalConditionedDiscreteGridWorld(map_name=settings.map_tag.value)
    elif settings.environment_tag == EnvironmentIndex.POINT_MAZE:
        environment = GoalConditionedPointEnv(map_name=settings.map_tag.value)
    elif settings.environment_tag == EnvironmentIndex.ANT_MAZE:
        environment = AntMaze(maze_name=settings.map_tag.value)
    else:
        raise NotImplementedError("Unknown environment type")

    #  '-> So we can test our agent at any time in a parallel environment, even in the middle of an episode
    test_agent.under_test = True
    test_agent.stop_episode()

    results = []
    goals = []
    reached_goals = []
    failed_goals = []
    for test_id in range(settings.nb_tests_per_evaluation):
        result, goal = eval_episode(settings, test_agent, environment)
        results.append(result)
        goals.append(goal)
        if result == 1:
            reached_goals.append(goal)
        else:
            failed_goals.append(goal)

    simulation_information.evaluation_stopwatch.stop()
    return mean(results) if results else None, goals, results


def eval_episode(settings: Settings, agent, environment):
    """
    Test the agent over a single goal reaching task. Return the result that will be directly passed to the DataHolder.
    return tuple(the closest node distance from goal, success in {0, 1})
    """
    state, goal = environment.reset()  # Reset our environment copy
    agent.start_episode(state, goal=goal, test_episode=True)
    agent.nb_successes_on_edges = 0
    agent.nb_failures_on_edges = 0

    reached = False
    test_duration = 0

    while not episode_done(settings, agent, test_duration, reached=reached, test_episode=True):
        action = agent.action(state)
        state, _, reached = environment.step(action)
        agent.process_interaction(action, None, state, None, learn=False)

        test_duration += 1
    agent.stop_episode()

    return float(reached), goal  # , images


"""
Miscellaneous
"""
def episode_done(settings: Settings, agent, interaction_id, reached=False, test_episode=False):
    """
    Test if the episode should be done according to the given parameters.
    """
    if isinstance(agent, RGL) or isinstance(agent, REO_RGL) or (isinstance(agent, SORB) and test_episode):
        # NB: RGL and TC_RGL are subclasses of RGL
        return reached or agent.done
    elif isinstance(agent, GoalConditionedAgent) or (isinstance(agent, SORB) and not test_episode):
        return reached or interaction_id >= settings.control_only_agent_max_steps
    else:
        raise Exception("Unknown agent type.")


def save_goals_image(environment, image_id, goals, results, simulation_id):
    directory = os.path.dirname(os.path.abspath(__file__)) + "/outputs/goals_images/" + str(simulation_id) + "/"
    create_dir(directory)
    filename = "goals_" + str(image_id)

    image = environment.render()
    for goal, reached in zip(goals, results):
        x, y = environment.get_coordinates(goal)
        image = environment.set_tile_color(image, x, y, [0, 255, 0] if reached else [255, 0, 0])

    save_image(image, directory, filename)


def generate_graph_image(env, graph, directory, file_name, pruned_edges=None, exploration_trajectory=None,
                         sampled_goal=None,
                         reached_goals=None, failed_goals=None,
                         node_from=None, exploration_node=None, exploration_goal=None):
    # Build image
    image = env.render(ignore_goal=True)

    if reached_goals is not None:
        for goal in reached_goals:
            env.place_point(image, goal, [0, 255, 0], width=5)
    if failed_goals is not None:
        for goal in failed_goals:
            env.place_point(image, goal, [255, 0, 0], width=5)

    # Fill image
    #  - Build nodes
    for node_id, attributes in graph.nodes(data=True):
        env.place_point(image, attributes["state"], [125, 255, 0], width=5)

    #  - Build edges
    for node_1, node_2, attributes in graph.edges(data=True):
        node_1_state = graph.nodes[node_1]["state"]
        node_2_state = graph.nodes[node_2]["state"]
        mid = (node_1_state + node_2_state) / 2

        try:
            cost = graph.get_edge_data(node_1, node_2, "cost")["cost"]
            color = [255, 0, 0] if cost == float("inf") else [0, 255, 0]
        except Exception as e:
            color = [255, 153, 0]

        env.place_edge(image, node_1_state, mid, color, width=5)

        try:
            cost = graph.get_edge_data(node_2, node_1, "cost")["cost"]
            color = [255, 0, 0] if cost == float("inf") else [0, 255, 0]
        except Exception as e:
            color = [255, 153, 0]
        env.place_edge(image, mid, node_2_state, color, width=5)

    if pruned_edges is not None:
        for node_1, node_2 in pruned_edges:
            nodes_states = nx.get_node_attributes(graph, "state")
            state_1 = nodes_states[node_1]
            state_2 = nodes_states[node_2]
            color = [255, 0, 0]
            env.place_edge(image, state_1, state_2, color, width=5)

    # Place optional points if they are given (to observe what the exploration trajectory looks like for example)
    if exploration_trajectory is not None:
        for exploration_state in exploration_trajectory:
            env.place_point(image, exploration_state, [0, 0, 180], width=5)
    if sampled_goal is not None and env.state_space.contains(sampled_goal.astype(env.state_space.dtype)):
        env.place_point(image, sampled_goal, [100, 100, 100], width=5)

    if node_from is not None:
        env.place_point(image, node_from, [255, 153, 100], width=7)
    if exploration_node is not None:
        env.place_point(image, exploration_node, [153, 51, 255], width=7)
    if exploration_goal is not None:
        env.place_point(image, exploration_goal, [255, 0, 0], width=7)

    # Save image
    create_dir(directory)  # Make sure the directory exists
    save_image(image, directory, file_name)

def generate_learning_graph_image(env, agent):
    # Build image
    image = env.render()
    graph = agent.reachability_graph
    # Fill image
    #  - Build nodes
    for node_id, attributes in graph.nodes(data=True):
        if isinstance(agent, RGL) and len(agent.global_path) > 1 and node_id == agent.global_path[-2]:
            env.place_point(image, attributes["state"], [125, 255, 0], width=5)
        else:
            env.place_point(image, attributes["state"], [125, 255, 0], width=5)

    #  - Build edges
    for node_1, node_2, attributes in graph.edges(data=True):
        try:
            cost = attributes["cost"]
            if graph.has_edge(node_2, node_1):
                cost = max(cost, graph.get_edge_data(node_2, node_1, "cost")["cost"])

            color = [255, 0, 0] if cost == float("inf") else [0, 255, 0]
        except:
            color = [0, 255, 0]
        env.place_edge(image, graph.nodes[node_1]["state"], graph.nodes[node_2]["state"], color, width=5)

    return image


if __name__ == "__main__":
    for i in range(10):
        # Setup settings for test

        settings = Settings(environment_tag=EnvironmentIndex.POINT_MAZE, agent_tag=AgentsIndex.RGL,
                            map_tag=MapsIndex.JOIN_ROOMS, simulation_id=i, pre_train_in_playground=False)
        # settings.nb_interactions_before_evaluation = 5000
        # settings.nb_episodes_before_evaluation = 10
        # settings.nb_tests_per_evaluation = 50
        # settings.nb_interactions_max = 500000
        # settings.pre_train_nb_episodes = 100
        # settings.simulation_name = "test"
        settings.redirect_std_output = True

        # cProfile.run("simulation(test_settings)")
        simulation(settings)
    print()
