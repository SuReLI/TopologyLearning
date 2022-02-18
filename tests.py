from random import choice

import numpy as np

from agents import TopologyLearner, TopologyLearnerMode
from environments.grid_world import DiscreteGridWorld
from settings import settings
from utils.data_holder import DataHolder
from utils.sys_fun import generate_video, save_image


def evaluation(simulation, environment, evaluation_id):
    agent = simulation.agent
    assert isinstance(agent, TopologyLearner)
    data_holder: DataHolder = simulation.data_holder
    data_holder.on_evaluation_start()
    selected_goals = ([], [])
    for test_id in range(settings.nb_tests):
        directory = agent.output_dir + "tests/eval_" + str(evaluation_id) + "/"
        filename = "test_" + str(test_id)
        data_holder.on_test(test(directory, filename, simulation, environment, selected_goals))
    data_holder.on_evaluation_end()
    return selected_goals


def get_test_image(environment, agent: TopologyLearner, goal):
    if not isinstance(environment, DiscreteGridWorld):
        return environment.render()
    image = environment.render()
    oracle = environment.get_oracle()

    if agent.next_node_way_point is not None:
        for state in oracle:
            node = agent.get_node_for_state(state)
            if node == agent.next_node_way_point:
                x, y = environment.get_coordinates(state)
                image = environment.set_tile_color(image, x, y, [196, 185, 137])

    x, y = environment.get_coordinates(goal)
    image = environment.set_tile_color(image, x, y, [0, 255, 0])

    x, y = environment.get_coordinates(agent.next_sub_goal)
    image = environment.set_tile_color(image, x, y, [0, 0, 255])

    x, y = environment.agent_coordinates
    image = environment.set_tile_color(image, x, y, [255, 0, 0])

    return image


def test(directory, filename, simulation, environment, selected_goals: tuple, video=True) -> tuple:
    """
    Test the agent over a single goal reaching task. Return the result that will be directly passed to the DataHolder.
    return tuple(the closest node distance from goal, success in {0, 1})
    """
    agent = simulation.agent
    assert isinstance(agent, TopologyLearner)
    if video:
        images_directory = directory + filename.split(".")[0] + "_images/"
        images = []
    oracle = environment.get_oracle()
    goal = choice(oracle)
    state = environment.reset()
    agent.on_episode_start(state, TopologyLearnerMode.GO_TO, 0, goal)
    closest_node_weights = agent.get_node_for_state(state, data=True)[1]["weights"]
    closest_node_distance = np.linalg.norm(closest_node_weights - goal, 2)

    if video:
        if isinstance(environment, DiscreteGridWorld):
            image = get_test_image(environment, agent, goal)
        images.append(image)
        save_image(image, images_directory, str(0) + ".png")

    for time_step_id in range(99999):
        action = agent.action(state)
        state, _, _, _ = environment.step(action)
        if video:
            image = get_test_image(environment, agent, goal)
            images.append(image)
            save_image(image, images_directory, str(time_step_id + 1) + ".png")
        agent.on_action_stop(action, state, None, None, train_policy=False, learn_topology=False)
        reached = agent.reached(state, goal, allow_area=False)
        if reached or agent.done:
            agent.on_episode_stop()
            if video:
                generate_video(images, directory, filename)
            if reached:
                selected_goals[0].append(goal)
                return closest_node_distance, 1
            elif agent.done:  # The agent consider he failed.
                selected_goals[1].append(goal)
                return closest_node_distance, 0
    if video:
        generate_video(images, directory, filename)
    raise Exception("Maximum time steps reached for a test")
