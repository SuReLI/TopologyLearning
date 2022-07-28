"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import time
from random import choice

import numpy as np

from old.src.environments import GoalConditionedDiscreteGridWorld
from old.src.settings import settings
from statistics import mean
from matplotlib import pyplot as plt
from old.src.agents.grid_world.graph_free import DqnHerDiffAgent
from old.src.utils.plots import init_plots

# Initialise data memories
last_seeds_train_accuracy_memories = []
current_seed_train_accuracy_memory = []
current_seed_train_results = []
last_seeds_test_accuracy_memories = []
current_seed_test_accuracy_memory = []
sampled_goals = []


def simulation(verbose=True):
    global current_seed_train_accuracy_memory, current_seed_train_results
    global environment, agent, sampled_goals
    current_seed_train_accuracy_memory = []
    current_seed_train_results = []
    agent.on_simulation_start()
    local_buffer = []
    for episode_id in range(settings.nb_episodes_per_simulation):
        time_step_id = 0
        state, _ = environment.reset()
        if episode_id != 0:
            goal = choice(local_buffer)
            sampled_goals.append(goal)
            agent.on_episode_start(state, goal)

        episode_rewards = []
        done = False
        while (not done) and time_step_id < settings.episode_length:

            if episode_id != 0:
                action = agent.action(state)
            else:
                action = environment.action_space.sample()
            state, reward, _ = environment.step(action)
            local_buffer.append(state)
            if episode_id != 0:
                if (state == goal).all():
                    done = True
                    reward = 1
                    current_seed_train_results.append(1)
                else:
                    reward = -1

            # Ending time learning_step process ...
            if episode_id != 0:
                agent.on_action_stop(action, state, reward, done, learn=True)

            # Store reward
            episode_rewards.append(reward)
            time_step_id += 1
        if episode_id != 0:
            if not done:
                current_seed_train_results.append(0)
            if len(current_seed_train_results) >= 20:
                current_seed_train_accuracy_memory.append(mean(current_seed_train_results[-20:]))
                if verbose:
                    print("Episode " + str(episode_id) + ", episode return " + str(current_seed_train_results[-1])
                          + ", last 20 avg " + str(current_seed_train_accuracy_memory[-1]))
            elif verbose:
                print("Episode " + str(episode_id) + ", episode return " + str(current_seed_train_results[-1]))
            agent.on_episode_stop()

        if episode_id != 0 and episode_id % settings.nb_episodes_before_tests == 0:
            test()

        # Plots if needed, for goals buffer representation, tests, and train accuracy
        if episode_id % settings.nb_episodes_before_plots == 0 and settings.nb_episodes_before_plots is not None:
            update_plots()


def test(verbose=False):
    global environment, agent
    global current_seed_test_accuracy_memory
    episodes_grades = []
    agent.on_simulation_start()

    # Initialise sampled goals plot (Fails in red and successes in blue)
    ax_test_goals.clear()  # Clear the last image if there is one, before to plot over it
    # Get the image representation of the environment as a list of pixels
    environment_bg_image = environment.get_environment_background(ignore_agent=True)

    # Run tests
    for test_id in range(settings.nb_tests):
        state, goal = environment.reset()
        goal_x, goal_y = environment.get_coordinates(goal)  # Used to plot the goal on the environment's bg image
        agent.on_episode_start(state, goal)

        time_step_id = 0
        while time_step_id < settings.episode_length:
            action = agent.action(state)
            state, reward, done = environment.step(action)
            if (state == goal).all():
                episodes_grades.append(1)
                environment_bg_image = environment.set_tile_color(environment_bg_image, goal_x, goal_y, [0, 0, 255])
                break
            time_step_id += 1
        else:
            episodes_grades.append(0)
            environment_bg_image = environment.set_tile_color(environment_bg_image, goal_x, goal_y, [255, 0, 0])

        agent.on_episode_stop()
        if verbose:
            print("Test " + str(test_id) + ", - result = " + str(episodes_grades[-1]) + ", avg "
                  + str(mean(episodes_grades)))
    current_seed_test_accuracy_memory.append(mean(episodes_grades))

    ax_test_goals.imshow(environment_bg_image)  # Plot image
    ax_test_goals.set_title("Goals sampled from oracle on tests")
    plt.show()
    plt.pause(.001)  # Mandatory to make the plots visible in interactive mode with program running in bg.


def update_plots(save_path=None):
    global agent
    global environment
    global ax_train_goals, ax_train_accuracy, ax_test_accuracy
    global sampled_goals, last_seeds_train_accuracy_memories, last_seeds_test_accuracy_memories,\
        current_seed_train_accuracy_memory, current_seed_test_accuracy_memory
    # Update others plots
    # Update sampled goals representation
    ax_train_goals.clear()
    img = environment.get_environment_background(ignore_agent=True)

    # For each sampled goal, place a blue tile on the image
    for goal in sampled_goals:
        x, y = environment.get_coordinates(goal)
        img = environment.set_tile_color(img, x, y, [0, 255, 0])
    ax_train_goals.imshow(img)  # Plot image

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

    # Update test accuracy graph
    ax_test_accuracy.clear()
    if seed_id != 0:
        data = np.array(last_seeds_test_accuracy_memories)
        means = np.mean(data, 0)
        stds = np.std(data, 0) if len(data > 0) else 0
        abscissa_values = [x * settings.nb_episodes_before_tests for x in range(1, len(means) + 1)]
        ax_test_accuracy.plot(abscissa_values, means, label="HER, old seeds")
        ax_test_accuracy.fill_between(abscissa_values, means + stds, means - stds, alpha=settings.std_area_transparency)
    if current_seed_test_accuracy_memory:
        abscissa_values = [x * settings.nb_episodes_before_tests for x in range(1, len(current_seed_test_accuracy_memory) + 1)]
        ax_test_accuracy.plot(abscissa_values, current_seed_test_accuracy_memory, "--", label="HER, current simulation")
        ax_test_accuracy.legend()

    ax_train_goals.set_title("Goals sampled for training from agent's memory")
    ax_train_accuracy.set_title("Accuracy of agent during training")
    ax_train_accuracy.set_xlabel("episodes")
    ax_train_accuracy.set_ylabel("accuracy")
    ax_test_accuracy.set_title("Agent's accuracy over tests")
    ax_test_accuracy.set_xlabel("episodes")
    ax_test_accuracy.set_ylabel("accuracy")

    plt.show()
    plt.pause(.001)

    if save_path is not None:
        plt.savefig(save_path)


# Prepare plots
fig, axs = init_plots(nb_rows=2, nb_cols=2)
fig.suptitle('Pre-train learning')
ax_train_goals, ax_train_accuracy, ax_test_goals, ax_test_accuracy = axs.flat

environment = GoalConditionedDiscreteGridWorld(map_id=7)

dqn_seeds_rewards_result = []
dqn_seeds_tests_result = []
colors = ["#ff0000", "#ff9500"]

agent = DqnHerDiffAgent(state_space=environment.observation_space, action_space=environment.action_space)

# for seed_id in range(settings.nb_seeds):
for seed_id in range(10):
    # print()
    # print("###################")
    # print()
    # print("      SEED " + str(seed_id))
    # print()
    # print("###################")

    simulation()

    last_seeds_train_accuracy_memories.append(current_seed_train_accuracy_memory)
    last_seeds_test_accuracy_memories.append(current_seed_test_accuracy_memory)
    current_seed_train_accuracy_memory = []
    current_seed_test_accuracy_memory = []

    if seed_id < settings.nb_seeds - 1:
        # Reset everything for next seed
        sampled_goals = []
        agent.reset()

update_plots(save_path="outputs/pre_train_1.png")
time.sleep(15)
