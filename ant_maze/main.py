"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import os
import pickle
from statistics import mean

import numpy as np
from matplotlib import pyplot as plt

from ant_maze.control_policy.agent import AntMazeControlPolicy
from utils.sys_fun import create_dir, save_image, get_red_green_color
import local_settings
from ant_maze.environment import AntMaze
from ant_maze.planning_agents.tipp import TIPP
from ant_maze.planning_agents.topological_graph_planning_agent import TopologyLearnerMode, PlanningTopologyLearner


def generate_graph_image(env, network, directory, file_name):

    # Build image
    image = env.render()

    # Fill image
    #  - Build nodes
    for node_id, attributes in network.nodes(data=True):
        env.place_point(image, attributes["state"][:2], [125, 255, 0], width=30)

    #  - Build edges
    mini, maxi = None, None
    for _, _, attributes in network.edges(data=True):
        cost = attributes["cost"]
        if mini is None or mini > cost:
            mini = cost
        if maxi is None or maxi < cost:
            maxi = cost
    for node_1, node_2, attributes in network.edges(data=True):
        cost = attributes["cost"]
        color_value = 1 if maxi - mini == 0 else 1 - ((cost - mini) / (maxi - mini))
        color = get_red_green_color(color_value, hexadecimal=False)
        env.place_edge(image, network.nodes[node_1]["state"][:2], network.nodes[node_2]["state"][:2], color,
                       width=25)

    # Save image
    create_dir(directory)  # Make sure the directory exists
    save_image(image, directory, file_name)


def generate_test_graph_image(env, network, path, directory, file_name, goal):

    # Build image
    image = env.render()

    # Fill image
    #  - Build nodes
    for node_id, attributes in network.nodes(data=True):
        color = [0, 0, 255] if node_id in path else [125, 255, 0]
        env.place_point(image, attributes["state"], color, width=30)

    #  - Build edges
    for node_1, node_2, attributes in network.edges(data=True):
        env.place_edge(image, network.nodes[node_1]["state"], network.nodes[node_2]["state"], [125, 255, 0], width=25)

    env.place_point(image, goal, [255, 0, 0], width=30)

    # Save image
    create_dir(directory)  # Make sure the directory exists
    save_image(image, directory, file_name)


def reset_ant_maze(ant_maze_environment):
    goal = ant_maze_environment.get_next_goal(test=True)
    state = ant_maze_environment.reset_sim(goal)
    return state, goal


def run_simulation(agent, environment):
    seed_evaluations_results = []
    agent.on_simulation_start()

    # Train
    interaction_id = 0
    evaluation_id = 0
    episode_id = 0

    while evaluation_id < local_settings.nb_evaluations_max:
        state, goal = environment.reset()
        print("Episode ", episode_id)

        if isinstance(agent, PlanningTopologyLearner):
            agent.on_episode_start(state, TopologyLearnerMode.LEARN_ENV)
        # TODO SORB
        # elif isinstance(agent, SORB):
        #     agent.on_episode_start(state, goal)
        else:
            agent.on_episode_start(state, None)

        while not agent.done:
            action = agent.action(state)
            state, _, reached = environment.step(action)
            interaction_id += 1
            agent.on_action_stop(action, state, None, None)

            # Evaluation if needed
            if interaction_id != 0 and interaction_id % local_settings.nb_interactions_before_evaluation == 0:
                # evaluation_start_time = datetime.now()
                seed_evaluations_results.append(evaluation(agent))
                evaluation_id += 1
                # pause_total_duration += (datetime.now() - evaluation_start_time)

                directory = os.path.dirname(__file__) + "/outputs/test_images/"
                generate_graph_image(environment, agent.topology, directory,
                                     "test_img_eval_" + str(evaluation_id) + ".png")
                a = 1

        episode_id += 1

        # Evaluation if needed
        if interaction_id != 0 and episode_id % local_settings.nb_episodes_before_evaluation == 0:
            # evaluation_start_time = datetime.now()
            seed_evaluations_results.append(evaluation(agent))
            evaluation_id += 1
            # pause_total_duration += (datetime.now() - evaluation_start_time)

            directory = os.path.dirname(__file__) + "/outputs/test_images/"
            generate_graph_image(environment, agent.topology, directory,
                                 "test_img_eval_" + str(evaluation_id) + ".png")
        print(seed_evaluations_results)

        print("Running success average in graph: ", agent.nb_successes_on_edges /
              (agent.nb_successes_on_edges + agent.nb_failures_on_edges) * 100)
        agent.on_episode_stop()
    # Stop simulation ...
    agent.on_simulation_stop()
    # end_time = datetime.now()
    save_seed_results(agent, seed_evaluations_results)
    print(seed_evaluations_results)


def evaluation(agent):
    # Get an agent copy and prepare it to the test
    test_agent = agent.copy()
    env = AntMaze(maze_name=local_settings.environment_map)
    #  '-> So we can test our agent at any time in a parallel environment, even in the middle of an episode

    if isinstance(test_agent, PlanningTopologyLearner):
        test_agent.on_episode_stop(learn=False)
    else:
        test_agent.on_episode_stop()

    results = []
    for test_id in range(local_settings.nb_tests_per_evaluation):
        result = test(test_agent, env)
        results.append(result)
    return mean(results)


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
    generate_test_graph_image(environment, agent.topology, agent.current_goal_reaching_nodes_path, directory,
                         "test_img_eval_" + str(0) + ".png", goal)

    reached = False
    test_duration = 0
    while not reached or not agent.done:
        action = agent.action(state)
        state, _, reached = environment.step(action)
        agent.on_action_stop(action, state, None, None, learn=False)
        test_duration += 1
        if reached or agent.done:
            agent.on_episode_stop()
            return float(reached)
    print("test: average accuracy on edges: ", agent.nb_successes_on_edges /
          (agent.nb_successes_on_edges + agent.nb_failures_on_edges) * 100)
    raise Exception("Maximum time steps reached for a test")


def save_seed_results(agent, results):
    # Find outputs directory
    # TODO: Verify directory
    seeds_outputs_directory = os.path.dirname(os.path.realpath(__file__)) + "outputs/seeds/" + agent.name + "/"
    create_dir(seeds_outputs_directory)

    # Get filename: Iterates through saved seeds to find an available id
    next_seed_id = 0  # Will be incremented for each saved seed we find.
    for filename in os.listdir(seeds_outputs_directory):
        if filename.startswith('seed_'):
            try:
                seed_id = int(filename.replace("seed_", ""))
            except:
                continue
            next_seed_id = seed_id + 1
    new_seed_directory = seeds_outputs_directory + "seed_" + str(next_seed_id) + "/"
    create_dir(new_seed_directory)

    with open(new_seed_directory + 'seed_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    """
    To load the results list, use:
    with open('parrot.pkl', 'rb') as f:
        mynewlist = pickle.load(f)
    """


def init():
    environment = AntMaze(maze_name=local_settings.environment_map, show=False)
    # environment.reset()
    low_policy = AntMazeControlPolicy(environment)
    agent = TIPP(state_space=environment.observation_space, action_space=environment.action_space,
                 sub_goal_thresholds=environment.goal_thresholds[:2], random_exploration_duration=700,
                 max_steps_to_reach=40,
                 goal_reaching_agent=low_policy, verbose=False, edges_similarity_threshold=0.5,
                 nodes_similarity_threshold=0.7)
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


def main():
    agent, environment = init()
    run_simulation(agent, environment)


if __name__ == "__main__":
    main()
