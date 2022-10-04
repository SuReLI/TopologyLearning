"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import copy
import math
import os
import pickle
import sys
from statistics import mean

import numpy as np
from matplotlib import pyplot as plt
import settings
from ant_maze.environment import AntMaze
from utils.sys_fun import create_dir, save_image, generate_video, get_red_green_color
import local_settings
from local_settings import MapsIndex
from agents import SORB, PlanningTopologyLearner, TopologyLearnerMode, STC, RGL
from ant_maze.control_policy.agent import AntMazeControlPolicy


from utils.stopwatch import Stopwatch

samples_stopwatch = Stopwatch()
# sample_stopwatch measure the total duration of every interaction run by the environment, in order to compute the
# average sample complexity at the end.
training_stopwatch = Stopwatch()
# The last one is used to know how long the training time is, without tests and samples complexity.

def generate_graph_image(env, network, directory, file_name):

    # Build image
    image = env.render()

    # Fill image
    #  - Build nodes
    for node_id, attributes in network.nodes(data=True):
        env.place_point(image, attributes["state"], [125, 255, 0], width=30)

    #  - Build edges
    for node_1, node_2, attributes in network.edges(data=True):
        color = [255, 0, 0] if attributes["cost"] == float("inf") else [0, 255, 0]
        env.place_edge(image, network.nodes[node_1]["state"], network.nodes[node_2]["state"], color, width=25)

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

def episode_done(agent, interaction_id, reached=False):
    """
    Test if the episode should be done according to the given parameters:
    :param agent: The agent we are training / testing
    :param interaction_id: how many interactions has been made so far in the episode
    :param reached: If the goal has been reached (for test episodes only, default value is set to false for training
        episodes).
    :return: boolean, True if the episode is done
    """
    if isinstance(agent, PlanningTopologyLearner) or isinstance(agent, SORB):
        # NB: RGL and STC are subclasses of PlanningTopologyLearner
        result = reached or agent.done
        return result
    elif isinstance(agent, AntMazeControlPolicy):
        return reached or interaction_id >= local_settings.her_max_steps
    else:
        raise Exception("Unknown agent type.")

def run_simulation(agent, environment, seed_id):
    global training_stopwatch, samples_stopwatch
    training_stopwatch.reset()
    samples_stopwatch.reset()  # Will be started everytime we ask the environment to compute agent's position.
    training_stopwatch.start()

    seed_evaluations_results = []
    agent.on_simulation_start()

    directory = os.path.dirname(__file__) + "/outputs/test_images/" + str(seed_id) + "/"
    generate_graph_image(environment, agent.topology, directory,
                         "test_img_eval_" + str("beg") + ".png")

    # Train
    interaction_id = 0
    evaluation_id = 0
    episode_id = 0

    while evaluation_id < local_settings.nb_evaluations_max:
        training_stopwatch.stop()
        samples_stopwatch.start()
        state, goal = environment.reset()
        samples_stopwatch.stop()
        training_stopwatch.start()
        advancement = interaction_id / (local_settings.nb_interactions_before_evaluation
                                    * local_settings.nb_evaluations_max) * 100
        print("Seed ", seed_id, ", episode ", episode_id, ", advancement: ", advancement, "%", sep='', end="\r")

        if isinstance(agent, PlanningTopologyLearner):
            agent.on_episode_start(state, TopologyLearnerMode.LEARN_ENV)
        else:
            agent.on_episode_start(state, goal)

        local_interaction_id = 0
        while not episode_done(agent, local_interaction_id):
            action = agent.action(state)

            training_stopwatch.stop()
            samples_stopwatch.start()
            state, reward, done = environment.step(action)
            samples_stopwatch.stop()
            training_stopwatch.start()
            local_interaction_id += 1
            interaction_id += 1
            if isinstance(agent, PlanningTopologyLearner):
                agent.on_action_stop(action, state, None, None)
            else:
                agent.on_action_stop(action, state, reward, done)

            # Evaluation if needed
            if interaction_id != 0 and interaction_id % local_settings.nb_interactions_before_evaluation == 0:
                training_stopwatch.stop()
                # evaluation_start_time = datetime.now()
                result, goals, results = evaluation(agent)
                seed_evaluations_results.append(result)
                evaluation_id += 1

                directory = os.path.dirname(__file__) + "/outputs/goals_image/"
                create_dir(directory)
                image = environment.render()
                for goal, result in zip(goals, results):
                    environment.place_point(image, goal, [0, 255, 0] if result else [255, 0, 0])
                save_image(image, directory, "goals_" + str(evaluation_id) + ".png")

                directory = os.path.dirname(__file__) + "/outputs/graph_images/" + str(seed_id) + "/"
                create_dir(directory)
                generate_graph_image(environment, agent.topology, directory,
                                     "test_img_eval_" + str(evaluation_id) + ".png")

                training_stopwatch.start()
        episode_id += 1
        agent.on_episode_stop()
    training_stopwatch.stop()

    seed_information = {
        "learning_time": training_stopwatch.get_duration(),
        "samples_average_duration": samples_stopwatch.get_duration() / interaction_id,
        "nb_total_interactions": interaction_id
    }
    if isinstance(agent, PlanningTopologyLearner) or isinstance(agent, SORB):
        seed_information["nodes_in_graph"] = len(agent.topology.nodes())
        seed_information["edges_in_graph"] = len(agent.topology.edges())
        seed_information["graph"] = agent.topology

    print(end="\x1b[2K")
    print("Seed ", seed_id, " advancement: Done.", sep='')
    print("accuracy_evolution = ", seed_evaluations_results, sep='')

    # Stop simulation ...
    agent.on_simulation_stop()
    return seed_evaluations_results, seed_information


def evaluation(agent):
    # Get an agent copy and prepare it to the test
    if isinstance(agent, PlanningTopologyLearner) or isinstance(agent, SORB):
        test_agent = copy.deepcopy(agent)
    else:
        test_agent = agent.copy()
    environment = AntMaze(maze_name=local_settings.map_name, show=False)
    #  '-> So we can test our agent at any time in a parallel environment, even in the middle of an episode

    if isinstance(test_agent, PlanningTopologyLearner):
        test_agent.on_episode_stop(learn=False)
    else:
        test_agent.on_episode_stop()

    results = []
    goals = []
    for test_id in range(local_settings.nb_tests_per_evaluation):
        result, goal = eval_episode(test_agent, environment)
        results.append(result)
        goals.append(goal)
    return mean(results), goals, results


def eval_episode(agent, environment):
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
    while not episode_done(agent, test_duration, reached=reached):
        action = agent.action(state)
        state, _, reached = environment.step(action)
        agent.on_action_stop(action, state, None, None, learn=False)
        test_duration += 1
    agent.on_episode_stop()
    return float(reached), goal


def save_seed_results(environment_name, agent, results, seed_information):
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

    with open(new_seed_directory + 'seed_info.pkl', 'wb') as f:
        pickle.dump(seed_information, f)
    with open(new_seed_directory + 'seed_abscissa.pkl', 'wb') as f:
        pickle.dump([(i + 1) * local_settings.nb_interactions_before_evaluation for i in range(len(results))], f)
    with open(new_seed_directory + 'seed_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(" >> seed saved in directory ", new_seed_directory)


def pre_train_gc_agent(environment, agent, nb_episodes=400, time_steps_max_per_episode=200):
    global training_stopwatch
    print("Pretraining low level agent ... please wait a bit ...")
    pre_training_agent = agent.goal_reaching_agent if isinstance(agent, PlanningTopologyLearner) else agent
    reached_goals = []
    pre_training_agent.on_simulation_start()
    results = []
    start_state = None
    if isinstance(agent, STC):
        last_trajectory = []
    for episode_id in range(nb_episodes):
        time_step_id = 0
        training_stopwatch.stop()
        state, goal = environment.reset()
        training_stopwatch.start()
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
            training_stopwatch.stop()
            state, reward, reached = environment.step(action)
            training_stopwatch.start()
            if isinstance(agent, STC):
                last_trajectory.append(state)
            done = reached or time_step_id > time_steps_max_per_episode
            pre_training_agent.on_action_stop(action, state, reward, done, learn=True)
            time_step_id += 1

        training_stopwatch.stop()
        results.append(1 if reached else 0)
        if len(results) > 20:
            last_20_average = mean(results[-20:])
        else:
            last_20_average = mean(results)
        print("Episode ", episode_id, " average result over last 20 episodes is ", last_20_average, "    ", end="\r")
        if episode_id == nb_episodes - 2:
            print(end="\x1b[2K")
        training_stopwatch.start()
        pre_training_agent.on_episode_stop()

        if isinstance(agent, STC):
            agent.store_tc_training_samples(last_trajectory)
            last_trajectory = []
    training_stopwatch.stop()
    return start_state, reached_goals


def init():
    environment = AntMaze(maze_name=local_settings.map_name, show=False)
    low_policy = AntMazeControlPolicy(environment, state_space=environment.observation_space,
                                      action_space=environment.action_space,
                                      device=settings.device)

    agents = [
        # RGL
        RGL(state_space=environment.observation_space, action_space=environment.action_space, tolerance_radius=0.3,
            random_exploration_duration=900, max_steps_to_reach=80, goal_reaching_agent=low_policy, verbose=False,
            edges_distance_threshold=1, nodes_distance_threshold=0.8, choose_exploration_target=False,
            exploration_goal_range=4, nb_explorations=1)
    ]
    """
        ,

        # SORB
        SORB(state_space=environment.observation_space, action_space=environment.action_space,
                     tolerance_radius=0.2, verbose=False, edges_distance_threshold=0.2, nb_nodes=400,
                     max_interactions_per_edge=40, max_final_interactions=50, environment=environment),

        # TI-STC
        STC(state_space=environment.observation_space, action_space=environment.action_space, tolerance_radius=0.1,
            random_exploration_duration=100, max_steps_to_reach=40, re_usable_policy=True,
            goal_reaching_agent=low_policy, verbose=False, edges_similarity_threshold=0.65,
            nodes_similarity_threshold=0.8, oriented_graph=False, translation_invariant_tc_network=True, name="TI-STC")
    """
    return agents, environment


def save_goals_image(environment, image_id, goals, results, seed_id):
    directory = os.path.dirname(__file__) + "/outputs/goals_images/" + str(seed_id) + "/"
    create_dir(directory)
    filename = "goals_" + str(image_id)

    image = environment.render()
    for goal, reached in zip(goals, results):
        x, y = environment.get_coordinates(goal)
        image = environment.set_tile_color(image, x, y, [0, 255, 0] if reached else [255, 0, 0])

    save_image(image, directory, filename)

def main():
    global training_stopwatch, samples_stopwatch
    agents, environment = init()
    start_state, _ = environment.reset()

    # Load pre_training information
    models_directory = os.path.dirname(__file__) + "/control_policy/models/"
    with open(models_directory + "reached_sub_goals.pkl", 'rb') as f:
        reached_goals = pickle.load(f)

    for agent in agents:
        """
        r = 11.7
        gran = r / 20
        points = []
        values = []
        initial_qpos = np.array([0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        state = np.concatenate((np.array([0., 0.]), initial_qpos, np.zeros(14)))
        for i in range(40):
            for j in range(40):
                a = (i - 19.5) * gran
                b = (j - 19.5) * gran
                points.append([a, b])
                goal = np.concatenate((np.array([a, b]), state[2:5]))

                # Compute angle to the goal.

                x = (i - 19.5) / 20
                y = (j - 19.5) / 20
                angle = math.acos(x) if y > 0 else -math.acos(x)
                state[6] = angle / math.pi
                agent.goal_reaching_agent.current_goal = goal
                action = agent.goal_reaching_agent.action(state)
                values.append(agent.goal_reaching_agent.layers[0].critic.get_Q_value(state[np.newaxis], goal[np.newaxis], action[np.newaxis]))

        mini = min(values)
        maxi = max(values)
        colors = [get_red_green_color((v - mini) / (maxi - mini), hexadecimal=False) for v in values]
        points = np.array(points)
        colors = np.array(colors) / 255
        plt.scatter(points[:, 0], points[:, 1], c=colors)
        plt.show()
        """
        for seed_id in range(5):
            print("#################")
            print("seed " + str(seed_id))
            print("#################")
            training_stopwatch.start()
            if isinstance(agent, SORB):
                vertices = [environment.sample_reachable_state() for i in range(agent.nb_nodes)]
                agent.on_pre_training_done(start_state, reached_goals, vertices)
            elif isinstance(agent, PlanningTopologyLearner):
                agent.on_pre_training_done(start_state, reached_goals)
            save_seed_results(local_settings.map_name, agent, *run_simulation(agent, environment, seed_id))
        print("\n")

if __name__ == "__main__":
    main()
