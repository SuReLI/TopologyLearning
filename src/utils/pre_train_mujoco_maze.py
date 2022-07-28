import os
from random import choice
from statistics import mean
from gym.spaces import Box
import numpy as np
from matplotlib import pyplot as plt
from typing import List

from src.agents.grid_world.graph_planning.stc import STC_TL
from src.agents.grid_world.graph_planning.topological_graph_planning_agent import PlanningTopologyLearner
from src.settings import settings
from src.utils.point_maze import reset_point_maze
from src.utils.sys_fun import generate_video

REACHED_GOALS = []
LOGS_MEANS_MEMORY = []
init_state = None
S_G_MEM_S = []


def sample_pre_train_target_state(environment, goal_extractor) -> np.ndarray:
    """
    This function is a default version. The used one will be given to the pre_train method.
    Sample a state that can be targeted during pre_training.

    :parameter environment: Environment the agent is inside.
    :parameter goal_extractor: function that extract a goal from a state.
    """
    assert callable(goal_extractor)

    # Sample a state to reach
    goal_position = Box(np.array([environment.reset_location[0] - 0.63, environment.reset_location[1] - 0.63]),
                        np.array([environment.reset_location[0] + .22, environment.reset_location[1] + .22])).sample()
    goal_angle = np.random.randn(2) * 10 - 5
    goal_velocity = np.random.randn(2) * 10 - 5
    target_state = np.concatenate((goal_position, goal_velocity))

    # Extract the goal from the generated state and return it
    return goal_extractor(target_state)


def reached(state, goal):
    return np.linalg.norm(state[:2] - goal[:2], 2) < settings.point_maze_reach_distance_threshold


def pre_train(agent, environment, state_to_goal_filter: List[int] = None,
              goal_sampler=sample_pre_train_target_state, reach_criteria=reached):
    """
    Pretrain the agent goal-conditioned policy using the following parameters
    :param agent: Agent to pre_train
    :param environment: Environment the agent is inside
    :param state_goal_scale: A list of real numbers, chosen empirically, used to scale the variables inside the state - goal
        tensor. We assume that every variable do not have the same importance in the verification of the reach
        condition.
    :param state_to_goal_filter: A list of 0 and 1 indicating if an element of the state should be kept when it is
    converted into a goal (1) or removed (0). Precondition: len(state_to_goal_filter) == len(state).
    :param goal_sampler: A FUNCTION that generate a goal, using the environment and the goal extractor. f: e, g -> G
    :param reach_criteria: A FUNCTION that verify if a given goal has been reached by a given state, re: s, g -> bool
    :return: None
    """

    type = environment.env.spec.entry_point.split(".")[1].split(":")[0]
    PRE_TRAIN_DELTA = []
    print("Pre-training agent ...")

    # Set state_to_goal_filter to the default value if needed, and convert is to the right type.
    if state_to_goal_filter is None:
        state_to_goal_filter = [1, 1, 0, 0]
    state_to_goal_filter = np.array(state_to_goal_filter).astype(np.bool)

    reached_goals = []
    if isinstance(agent, PlanningTopologyLearner):
        _agent = agent.goal_reaching_agent
    else:
        _agent = agent
    _agent.on_simulation_start()

    # Used observed states to train the policy
    initial_state = None

    last_average_rewards = []
    avg_last_average_rewards = []

    # Episode images storage
    gen_videos = True
    nb_videos = 10  # IMAGE
    images = []

    if type == "pointmaze":
        nb_episodes = settings.point_maze_pre_training_duration
    else:
        nb_episodes = settings.ant_maze_pre_training_duration
    for episode_id in range(nb_episodes):
        directory = "outputs/pre_train_videos/"
        time_step_id = 0
        if type == "pointmaze":
            state, _ = environment.reset()
        else:
            state = environment.reset()

        if initial_state is None:
            initial_state = state
        if isinstance(agent, STC_TL):
            # Store samples for TC-Network training
            agent.last_episode_trajectory = [state]
        goal = goal_sampler(environment, state_to_goal_filter)

        # IMAGES
        if gen_videos:
            image = environment.get_background_image()
            image = environment.state_on_image(goal, image=image, color=[0, 255, 0])
            image = environment.state_on_image(state, image=image, color=[0, 0, 255])
            images.append(image)

        delta = state[state_to_goal_filter] - goal
        for d in PRE_TRAIN_DELTA:
            if (d == delta).all():
                break
        else:
            PRE_TRAIN_DELTA.append(delta)

        _agent.on_episode_start(state, goal)
        done = False

        if type == "pointmaze":
            episode_duration = settings.point_maze_episode_length
        else:
            episode_duration = settings.ant_maze_episode_length
        while (not done) and time_step_id < episode_duration:
            action = _agent.action(state)
            returned = environment.step(action)
            state = returned[0]

            # IMAGES
            if gen_videos:
                image = environment.get_background_image()
                image = environment.state_on_image(goal, image=image, color=[0, 255, 0])
                image = environment.state_on_image(state, image=image, color=[0, 0, 255])
                images.append(image)

            if reach_criteria(state, goal):
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False

            # Ending time learning_step process ...
            _agent.on_action_stop(action, state, reward, done)
            time_step_id += 1
        assert done or time_step_id >= episode_duration

        # IMAGES: save video
        if gen_videos:
            generate_video(images, "../" + directory, str(episode_id % nb_videos) + ".mp4")
            images = []

        if isinstance(agent, STC_TL):
            # Train TC-Network
            agent.last_episode_trajectory.append(state)
            agent.train_tc_network()
        if done:
            if hasattr(agent, "on_pre_training_done") and callable(agent.on_pre_training_done):
                # Some algorithm need a list of reached goals to scale the states similarity.
                # If the current reached goal is not in the list, we add him.
                for elt in reached_goals:
                    if (elt == goal).all():
                        # The current reached goal is already in the list, let's skip this one.
                        break
                else:  # We fall here if the 'break' above hasn't been called.
                    reached_goals.append(goal)

        # Compute if the current goal has already been reached before or not
        last_average_rewards.append(1 if done else 0)
        if done:
            S_G_MEM_S.append(initial_state[state_to_goal_filter] - goal)
        if len(last_average_rewards) > 20:
            print("episode average reward = ", last_average_rewards[-1], " last 20 avg. : ",
                  mean(last_average_rewards[-20:]))
            avg_last_average_rewards.append(mean(last_average_rewards[-20:]))
        else:
            print("episode average reward = ", last_average_rewards[-1])

        if True:
            axes = plt.figure(plt.get_fignums()[0]).get_axes()

            # Plot s - g we trained on so far
            axes[2].cla()
            if S_G_MEM_S:
                data = np.array(S_G_MEM_S)
                axes[2].scatter(data[:, 0], data[:, 1])

            # Observe correlation between q_value and distance for each state in oracle
            axes[3].cla()
            distances = []
            q_values = []
            target = np.array([3., 2.])
            for i in range(15):
                for j in range(15):
                    state_to_test = np.zeros(environment.observation_space.shape)
                    state_to_test[:2] = np.array([i * 0.2 + .45, j * 0.2 + .45])
                    distances.append(np.linalg.norm(state_to_test[state_to_goal_filter] - target, 2))
                    q_values.append(_agent.get_q_value(state_to_test, target))
            axis = axes[3]
            axis.scatter(distances, q_values)
            plt.show()
            plt.pause(0.001)

        agent.episode_id += 1
        _agent.on_episode_stop()
        if isinstance(agent, STC_TL):
            agent.store_tc_training_samples()

    if hasattr(agent, "on_pre_training_done") and callable(agent.on_pre_training_done):
        agent.on_pre_training_done(initial_state, reached_goals)

    print("Pre-training is done.")
    return PRE_TRAIN_DELTA
