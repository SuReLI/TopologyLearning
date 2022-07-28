"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import copy
import time

import gym
import numpy as np
import d4rl
from src.settings import settings
from statistics import mean
import matplotlib.pyplot as plt
from src.agents import SACHERDiffAgent, SACHERAgent
from src.utils.plots import init_plots
import d4rl

# Initialise data memories
from src.utils.sys_fun import create_dir

last_seeds_train_accuracy_memories = []
current_seed_train_accuracy_memory = []
distances = []


def simulation(verbose=True):
    global current_seed_train_accuracy_memory
    global environment, agent, sampled_goals
    current_seed_train_accuracy_memory = []
    current_seed_train_results = []
    agent.on_simulation_start()
    # local_buffer = []
    for episode_id in range(settings.ant_maze_nb_episodes_per_simulation):
        state, goal = reset_environment()

        # print("state = ", state)
        # print("goal = ", goal)
        agent.on_episode_start(state, goal)

        done = False
        episode_rewards_sum = 0
        while not done:
            action = agent.action(state)
            state, reward, done, info = environment.step(action)
            # print("state = ", state[:2])
            sg_distance = np.linalg.norm(state[:2] - goal, 2)
            distances.append(sg_distance)
            episode_rewards_sum += reward
            # Ending time learning_step process ...
            # if episode_id != 0:
            agent.on_action_stop(action, state, reward, done, learn=True)

        current_seed_train_results.append(episode_rewards_sum)
        printed = False
        if len(current_seed_train_results) >= 20:
            current_seed_train_accuracy_memory.append(mean(current_seed_train_results[-20:]))
            if verbose:
                print("Episode " + str(episode_id) + ", episode return " + str(current_seed_train_results[-1])
                      + ", last 20 avg " + str(current_seed_train_accuracy_memory[-1]))
                printed = True
        elif verbose:
            print("Episode " + str(episode_id) + ", episode return " + str(current_seed_train_results[-1]))
            printed = True
        assert printed
        agent.on_episode_stop()

        # Plots if needed, for goals buffer representation, tests, and train accuracy
        if episode_id % settings.nb_episodes_before_plots == 0 and settings.nb_episodes_before_plots is not None:
            update_plots()


def update_plots(save_path=None):
    global agent, distances
    global ax_train_goals, ax_train_accuracy, ax_test_accuracy
    global sampled_goals, last_seeds_train_accuracy_memories, last_seeds_test_accuracy_memories,\
        current_seed_train_accuracy_memory, current_seed_test_accuracy_memory
    # Update others plots

    # Update train accuracy graph
    ax_train_accuracy.clear()

    if seed_id != 0:
        data = np.array(last_seeds_train_accuracy_memories)
        means = np.mean(data, 0)
        stds = np.std(data, 0) if len(data > 0) else 0
        abscissa_values = [20 + x for x in range(1, len(means) + 1)]
        ax_train_accuracy.plot(abscissa_values, means, label="HER, old seeds")
        ax_train_accuracy.fill_between(abscissa_values, means + stds, means - stds,
                                       alpha=settings.std_area_transparency)
    if current_seed_train_accuracy_memory:
        abscissa_values = [20 + x for x in range(1, len(current_seed_train_accuracy_memory) + 1)]
        ax_train_accuracy.plot(abscissa_values, current_seed_train_accuracy_memory, "--",
                               label="HER, current simulation")
        ax_train_accuracy.legend()

    ax_train_accuracy.set_title("Accuracy of agent during training")
    ax_train_accuracy.set_xlabel("episodes")
    ax_train_accuracy.set_ylabel("accuracy")

    ax_test_accuracy.set_title("distances")
    ax_train_accuracy.set_xlabel("episodes")
    ax_train_accuracy.set_ylabel("distance")
    ax_test_accuracy.plot(distances)

    plt.show()
    plt.pause(.001)

    if save_path is not None:
        directory = "".join(save_path.split("/")[:-1])[:-1]
        create_dir(directory)
        plt.savefig(save_path)


# Prepare plots
fig, axs = init_plots(nb_rows=2, nb_cols=2)
fig.suptitle('Pre-train learning')
_, ax_train_accuracy, _, ax_test_accuracy = axs.flat


"""
"antmaze-umaze-v2"
"""
environment = gym.make("maze2d-umaze-v1", reward_type="sparse", reset_target=True)


def reset_environment():
    global environment
    environment.reset()
    goal = environment.get_target()
    initial_state = environment.reset_to_location((3, 1))
    return initial_state, goal


dqn_seeds_rewards_result = []
dqn_seeds_tests_result = []
colors = ["#ff0000", "#ff9500"]

agent = SACHERAgent(state_space=environment.observation_space, action_space=environment.action_space)

# for seed_id in range(settings.nb_seeds):
for seed_id in range(10):
    print()
    print("###################")
    print()
    print("      SEED " + str(seed_id))
    print()
    print("###################")

    simulation()

    last_seeds_train_accuracy_memories.append(current_seed_train_accuracy_memory)
    current_seed_train_accuracy_memory = []

    if seed_id < settings.nb_seeds - 1:
        # Reset everything for next seed
        sampled_goals = []
        agent.reset()

update_plots(save_path="outputs/ant_maze/her.png")
time.sleep(15)
