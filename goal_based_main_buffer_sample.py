"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
from random import choice

import gym
import numpy as np
import environments
from settings import settings
from statistics import mean
from matplotlib import pyplot as plt
from agents.goal_conditioned_rl_agents.Discrete.dqn_her import DQNHERAgent
from agents.goal_conditioned_rl_agents.Discrete.sac_discrete.sac_her import SACHERAgent


def simulation(environment, agent, nb_episodes=100, verbose=True):
    time_steps_max_per_episode = 80
    episodes_rewards_sum = []
    test_results = []
    average_training_accuracy = []
    agent.on_simulation_start()
    local_buffer = []
    for episode_id in range(nb_episodes):
        training_accuracy = []
        time_step_id = 0
        state, _ = environment.reset()
        if episode_id != 0:
            goal = choice(local_buffer)
            agent.on_episode_start(state, goal, episode_id)

        episode_rewards = []
        done = False
        while (not done) and time_step_id < time_steps_max_per_episode:

            if episode_id != 0:
                action = agent.action(state)
            else:
                action = environment.action_space.sample()
            state, reward, _, _ = environment.step(action)
            local_buffer.append(state)
            if episode_id != 0:
                if (state == goal).all():
                    done = True
                    reward = 1
                    training_accuracy.append(1)
                else:
                    reward = -1

            # Ending time step process ...
            if episode_id != 0:
                agent.on_action_stop(action, state, reward, done)

            # Store reward
            episode_rewards.append(reward)
            time_step_id += 1
        if episode_id != 0:
            if not done:
                training_accuracy.append(0)
            average_training_accuracy.append(mean(training_accuracy))
            agent.on_episode_stop()
            rewards_sum = sum(episode_rewards)
            episodes_rewards_sum.append(rewards_sum)
            if len(average_training_accuracy) > 20:
                last_20_average = mean(average_training_accuracy[-20:])
            else:
                last_20_average = mean(average_training_accuracy)

            if verbose:
                print("Episode " + str(episode_id) + ", episode return " + str(average_training_accuracy[-1])
                      + ", last 20 avg " + str(last_20_average))
        environment.close()

        if episode_id != 0 and episode_id % 10 == 0:
            render = False
            test_results.append(test(environment, agent, verbose=False, render=render))

    return average_training_accuracy, test_results  # We will plot tests results, not the sum of rewards


def test(environment, agent, nb_tests=20, verbose=True, render=False):
    time_steps_max_per_episode = 80
    episodes_grades = []
    agent.on_simulation_start()
    average_grade = None
    for test_id in range(nb_tests):
        state, goal = environment.reset()
        agent.on_episode_start(state, goal, 0)
        if render:
            plt.cla()
            image = environment.render()
            plt.imshow(image)
            plt.show()

        time_step_id = 0
        while time_step_id < time_steps_max_per_episode:
            action = agent.action(state)
            state, reward, done, _ = environment.step(action)
            if (state == goal).all():
                episodes_grades.append(1)
            time_step_id += 1
        if time_step_id >= time_steps_max_per_episode:
            episodes_grades.append(0)

        # Compute distance to goal
        # distance_to_goal = np.linalg.norm(state[-2:] - goal, 2)
        # episodes_grades.append(distance_to_goal)
        agent.on_episode_stop()
        environment.close()
        average_grade = mean(episodes_grades)
        if verbose:
            print("Test " + str(test_id) + ", - result = " + str(episodes_grades[-1]) + ", avg "
                  + str(average_grade))
    return average_grade


environment = gym.make("goal_conditioned_discrete_grid_world-v0")
environment.reset_with_map_id(2)

nb_seeds = 4
nb_episodes_per_simulations = 600

dqn_seeds_rewards_result = []
dqn_seeds_tests_result = []
colors = ["#ff0000", "#ff9500"]

for seed_id in range(nb_seeds):
    print()
    print("###################")
    print()
    print("      SEED " + str(seed_id))
    print()
    print("###################")

    print()
    print(" > Training DQN")
    print()
    dqn_agent = DQNHERAgent(environment.observation_space, environment.action_space)
    rewards_sum_per_episode, tests_results = simulation(environment, dqn_agent,
                                                        nb_episodes=nb_episodes_per_simulations)
    dqn_seeds_rewards_result.append(rewards_sum_per_episode)
    dqn_seeds_tests_result.append(tests_results)

    plt.cla()
    plt.title("Rewards sum per episode")
    dqn_means = np.mean(np.array(dqn_seeds_rewards_result), axis=0)
    dqn_stds = np.std(np.array(dqn_seeds_rewards_result), axis=0)
    plt.plot(dqn_means, color=colors[0], label="DQN")
    plt.fill_between([x for x in range(len(dqn_means))], dqn_means + dqn_stds, dqn_means - dqn_stds, color=colors[0],
                     alpha=0.2)
    plt.legend()
    plt.show()

    plt.cla()
    plt.title("Tests grades evolution during agent learning")
    dqn_means = np.mean(np.array(dqn_seeds_tests_result), axis=0)
    dqn_stds = np.std(np.array(dqn_seeds_tests_result), axis=0)
    plt.plot(dqn_means, color=colors[0], label="DQN")
    plt.fill_between([x for x in range(len(dqn_means))], dqn_means + dqn_stds, dqn_means - dqn_stds, color=colors[0],
                     alpha=0.2)
    plt.legend()
    plt.show()
