"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import os
import pickle
from copy import deepcopy
from statistics import mean

import numpy as np
from matplotlib import pyplot as plt

import settings
from agents.graph_planning.stc import STC
from agents.graph_planning.rgl import RGL
from agents.graph_planning.topological_graph_planning_agent import PlanningTopologyLearner, TopologyLearnerMode
from utils.sys_fun import create_dir, save_image
import local_settings
from point_maze.environment import GoalConditionedPointMaze, MapsIndex
from agents import GoalConditionedSacHerAgent, GoalConditionedSacHerDiffAgent


def generate_graph_image(env, network, directory, file_name):

    # Build image
    image = env.render()

    # Fill image
    #  - Build nodes
    for node_id, attributes in network.nodes(data=True):
        env.place_point(image, attributes["state"], [125, 255, 0], width=5)

    #  - Build edges
    for node_1, node_2, attributes in network.edges(data=True):
        color = [255, 0, 0] if attributes["cost"] == float("inf") else [0, 255, 0]
        env.place_edge(image, network.nodes[node_1]["state"], network.nodes[node_2]["state"], color, width=5)

    # Save image
    create_dir(directory)  # Make sure the directory exists
    save_image(image, directory, file_name)


def generate_test_graph_image(env, network, path, directory, file_name, goal):

    # Build image
    image = env.get_background_image()

    # Fill image
    #  - Build nodes
    for node_id, attributes in network.nodes(data=True):
        color = [0, 0, 255] if node_id in path else [125, 255, 0]
        env.place_point(image, *tuple(attributes["state"][:2]), color, width=30)

    #  - Build edges
    for node_1, node_2, attributes in network.edges(data=True):
        env.place_edge(image, *tuple(network.nodes[node_1]["state"][:2]),
                       *tuple(network.nodes[node_2]["state"][:2]), [125, 255, 0], width=25)

    env.place_point(image, *goal[:2], [255, 0, 0], width=30)

    # Save image
    create_dir(directory)  # Make sure the directory exists
    save_image(image, directory, file_name)


def reset_ant_maze(ant_maze_environment):
    goal = ant_maze_environment.get_next_goal(test=True)
    state = ant_maze_environment.reset_sim(goal)
    return state, goal


def run_simulation(agent, environment, seed_id):
    seed_evaluations_results = []
    agent.on_simulation_start()

    # Train
    interaction_id = 0
    evaluation_id = 0
    episode_id = 0

    while evaluation_id < local_settings.nb_evaluations_max:
        state, goal = environment.reset()
        advancement = episode_id / (local_settings.nb_episodes_before_evaluation
                                    * local_settings.nb_evaluations_max) * 100
        print("Seed ", seed_id, ", episode ", episode_id, ", advancement: ", advancement, "%", sep='', end="\r")

        if isinstance(agent, PlanningTopologyLearner):
            agent.on_episode_start(state, TopologyLearnerMode.LEARN_ENV)
        # TODO SORB
        # elif isinstance(agent, SORB):
        #     agent.on_episode_start(state, goal)
        else:
            agent.on_episode_start(state, None)

        while not agent.done:
            action = agent.action(state)
            state, _, _ = environment.step(action)
            interaction_id += 1
            agent.on_action_stop(action, state, None, None)
        episode_id += 1

        # Evaluation if needed
        if interaction_id != 0 and episode_id % local_settings.nb_episodes_before_evaluation == 0:
            # evaluation_start_time = datetime.now()
            result, goals, results = evaluation(agent)
            seed_evaluations_results.append(result)
            evaluation_id += 1
            # pause_total_duration += (datetime.now() - evaluation_start_time)

            directory = os.path.dirname(__file__) + "/outputs/test_images/" + str(seed_id) + "/"
            generate_graph_image(environment, agent.topology, directory,
                                 "test_img_eval_" + str(evaluation_id) + ".png")
            save_goals_image(environment, evaluation_id, goals, results, seed_id)
        agent.on_episode_stop()
    print(end="\x1b[2K")
    print("Seed ", seed_id, " advancement: Done.", sep='')
    print("accuracy_evolution = ", seed_evaluations_results, sep='')

    # Stop simulation ...
    agent.on_simulation_stop()
    return seed_evaluations_results


def evaluation(agent):
    # Get an agent copy and prepare it to the test
    test_agent = deepcopy(agent)
    env = GoalConditionedPointMaze(map_name=local_settings.map_name)
    #  '-> So we can test our agent at any time in a parallel environment, even in the middle of an episode

    if isinstance(test_agent, PlanningTopologyLearner):
        test_agent.on_episode_stop(learn=False)
    else:
        test_agent.on_episode_stop()

    results = []
    goals = []
    for test_id in range(local_settings.nb_tests_per_evaluation):
        result, goal = test(test_agent, env)
        results.append(result)
        goals.append(goal)
    return mean(results), goals, results


def test(agent, environment):
    """
    Test the agent over a single goal reaching task. Return the result that will be directly passed to the DataHolder.
    return tuple(the closest node distance from goal, success in {0, 1})
    """
    state, goal = environment.reset()                                      # Reset our environment copy
    if isinstance(agent, PlanningTopologyLearner):
        agent.on_episode_start(state, TopologyLearnerMode.GO_TO, goal)     # reset our agent copy
    else:
        agent.on_episode_start(state, goal)
    agent.nb_successes_on_edges = 0
    agent.nb_failures_on_edges = 0

    # Generate
    directory = os.path.dirname(__file__) + "/outputs/test_images/test_path/"
    create_dir(directory)
    # generate_test_graph_image(environment, agent.topology, agent.current_goal_reaching_nodes_path, directory,
    #                      "test_img_eval_" + str(0) + ".png", goal)

    reached = False
    test_duration = 0
    while not reached or not agent.done:
        action = agent.action(state)
        state, _, reached = environment.step(action)
        agent.on_action_stop(action, state, None, None, learn=False)
        test_duration += 1
        if reached or agent.done:
            agent.on_episode_stop()
            return float(reached), goal
    raise Exception("Maximum time steps reached for a test")


def save_seed_results(environment_name, agent, results):
    # Find outputs directory
    seeds_outputs_directory = os.path.dirname(__file__) + "/outputs/seeds/" + environment_name + "/" \
                              + agent.name + "/"
    create_dir(seeds_outputs_directory)

    # Get filename: Iterates through saved seeds to find an available id
    next_seed_id = 0  # Will be incremented for each saved seed we find.
    for filename in os.listdir(seeds_outputs_directory):
        if filename.startswith('seed_'):
            try:
                seed_id = int(filename.replace("seed_", ""))
            except:
                continue
            next_seed_id = max(next_seed_id, seed_id + 1)
    new_seed_directory = seeds_outputs_directory + "seed_" + str(next_seed_id) + "/"
    create_dir(new_seed_directory)

    with open(new_seed_directory + 'seed_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(new_seed_directory + 'seed_abscissa.pkl', 'wb') as f:
        pickle.dump([(i + 1) * local_settings.nb_episodes_before_evaluation for i in range(len(results))], f)
    print(" >> seed saved in directory ", new_seed_directory)


def pre_train_gc_agent(environment, agent, nb_episodes=400, time_steps_max_per_episode=200):
    print("Pretraining low level agent ... please wait a bit ...")
    pre_training_agent = agent.goal_reaching_agent

    reached_goals = []
    pre_training_agent.on_simulation_start()
    results = []
    start_state = None
    if isinstance(agent, STC):
        last_trajectory = []
    running_mean_accuracies = []
    for episode_id in range(nb_episodes):
        time_step_id = 0
        state, goal = environment.reset()
        if isinstance(agent, STC):
            last_trajectory.append(state)
        # Ate the end of the pre-training, the agent will have an accuracy of 1 (100 %). Then, we add every goals to
        # the reached goals list, so we can have more goals, even if they hasn't been reached yet.
        reached_goals.append(goal)
        if start_state is None:
            start_state = state.copy()
        pre_training_agent.on_episode_start(state, goal)

        reached = False
        done = False
        while not done:
            action = pre_training_agent.action(state)
            state, reward, reached = environment.step(action)
            if isinstance(agent, STC):
                last_trajectory.append(state)
            done = reached or time_step_id > time_steps_max_per_episode
            pre_training_agent.on_action_stop(action, state, reward, done)
            time_step_id += 1
        results.append(1 if reached else 0)

        if len(results) > 20:
            last_20_average = mean(results[-20:])
        else:
            last_20_average = mean(results)
        running_mean_accuracies.append(last_20_average)
        print("Episode ", episode_id, " average result over last 20 episodes is ", last_20_average, "    ", end="\r")
        if episode_id == nb_episodes - 2:
            print(end="\x1b[2K")
        pre_training_agent.on_episode_stop()

        if isinstance(agent, STC):
            agent.store_tc_training_samples(last_trajectory)
            last_trajectory = []
    return start_state, reached_goals

def init():
    environment = GoalConditionedPointMaze(map_name=local_settings.map_name)

    low_policy = GoalConditionedSacHerDiffAgent(state_space=environment.state_space,
                                                action_space=environment.action_space,
                                                device=settings.device)
    agent = RGL(state_space=environment.state_space, action_space=environment.action_space, tolerance_radius=0.5,
                random_exploration_duration=200, max_steps_to_reach=80, goal_reaching_agent=low_policy, verbose=False,
                edges_similarity_threshold=0.3, nodes_similarity_threshold=0.1)
    """
    agent = STC_TL(observation_space=environment.observation_space, action_space=environment.action_space,
                            tolerance_margin=tiles_dimensions, random_exploration_duration=100)
    agent = SORB(observation_space=environment.observation_space, action_space=environment.action_space,
                          tolerance_margin=tiles_dimensions, random_exploration_duration=100
                          oracle=environment.get_oracle())
    agent = AutonomousDQNHERAgent(observation_space=environment.observation_space,
                                           action_space=environment.action_space)
    agent = SORB_NO_ORACLE(observation_space=environment.observation_space, action_space=environment.action_space,
                                    tolerance_margin=tiles_dimensions, random_exploration_duration=100)
    agent = TIPP_GWR(observation_space=environment.observation_space, action_space=environment.action_space,
                              tolerance_margin=tiles_dimensions, random_exploration_duration=100)
    """
    return agent, environment


def save_goals_image(environment, image_id, goals, results, seed_id):
    directory = os.path.dirname(__file__) + "/outputs/goals_images/" + str(seed_id) + "/"
    create_dir(directory)
    filename = "goals_" + str(image_id)

    image = environment.render()
    for goal, reached in zip(goals, results):
        environment.place_point(image, goal, [0, 255, 0] if reached else [255, 0, 0])
    save_image(image, directory, filename)

def main():
    for seed_id in range(10):
        print("#################")
        print("seed " + str(seed_id))
        print("#################")
        agent, environment = init()
        pre_train_environment = GoalConditionedPointMaze(map_name=MapsIndex.EMPTY.value)
        start_state, reached_goals = pre_train_gc_agent(pre_train_environment, agent,
                                                        nb_episodes=local_settings.pre_train_nb_episodes,
                                                        time_steps_max_per_episode=local_settings.pre_train_nb_time_steps_per_episode)
        agent.on_pre_training_done(start_state, reached_goals)
        save_seed_results(local_settings.map_name, agent, run_simulation(agent, environment, seed_id))

        print("\n")

if __name__ == "__main__":
    main()
