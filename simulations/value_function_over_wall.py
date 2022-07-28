"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import time
from random import choice

import numpy as np
import torch

from old.src.environments import GoalConditionedDiscreteGridWorld
from old.src.settings import settings
from statistics import mean
from matplotlib import pyplot as plt
from old.src.agents.grid_world.graph_free.goal_conditioned_dqn_her import DQNHERAgent
from old.src.utils.plots import init_plots

# Initialise data memories
from old.src.utils.sys_fun import get_red_green_color

last_seeds_train_accuracy_memories = []
current_seed_train_accuracy_memory = []
current_seed_train_results = []
sampled_goals = []


def simulation(verbose=True):
    global environment, agent
    global current_seed_train_accuracy_memory, current_seed_train_results

    goals_memory_max_size = 100

    # Here, we don't need to demonstrate anything. To make sure the training is efficient and trustworthy, we will use
    # and oracle that will help us to build a more efficient policy and value function.
    oracle = environment.get_oracle()

    current_seed_train_accuracy_memory = []
    current_seed_train_results = []
    agent.on_simulation_start()
    for episode_id in range(settings.nb_episodes_per_simulation):
        time_step_id = 0
        state, _ = environment.reset()
        if episode_id != 0:
            goal = choice(oracle)
            agent.on_episode_start(state, goal)

        episode_rewards = []
        done = False
        while (not done) and time_step_id < settings.episode_length:

            if episode_id != 0:
                action = agent.action(state)
            else:
                action = environment.action_space.sample()
            state, reward, _ = environment.step(action)
            if episode_id != 0:
                if (state == goal).all():
                    done = True
                    reward = 1
                    current_seed_train_results.append(1)
                    if len(sampled_goals) > goals_memory_max_size:
                        sampled_goals.pop(0)
                    sampled_goals.append([goal, 1])
                else:
                    reward = -1

            # Ending time learning_step process ...
            if episode_id != 0:
                agent.on_action_stop(action, state, reward, done)

            # Store reward
            episode_rewards.append(reward)
            time_step_id += 1
        if episode_id != 0:
            if not done:
                current_seed_train_results.append(0)
                if len(sampled_goals) > goals_memory_max_size:
                    sampled_goals.pop(0)
                sampled_goals.append([goal, 0])
            if len(current_seed_train_results) >= 20:
                current_seed_train_accuracy_memory.append(mean(current_seed_train_results[-20:]))
                if verbose:
                    print("Episode " + str(episode_id) + ", episode return " + str(current_seed_train_results[-1])
                          + ", last 20 avg " + str(current_seed_train_accuracy_memory[-1]))
            elif verbose:
                print("Episode " + str(episode_id) + ", episode return " + str(current_seed_train_results[-1]))
            agent.on_episode_stop()

        # Plots if needed, for goals buffer representation, tests, and train accuracy
        if episode_id % settings.nb_episodes_before_plots == 0 and settings.nb_episodes_before_plots is not None:
            update_plots()


def update_plots(save_path=None):
    global agent
    global environment
    global ax_train_goals, ax_train_accuracy, ax_test_accuracy
    global sampled_goals, last_seeds_train_accuracy_memories, current_seed_train_accuracy_memory
    # Update others plots
    # Update sampled goals representation
    ax_train_goals.clear()
    img = environment.get_environment_background(ignore_agent=True)

    # For each sampled goal, place a green tile on the image
    for goal, reached in sampled_goals:
        x, y = environment.get_coordinates(goal)
        if reached:
            img = environment.set_tile_color(img, x, y, [0, 0, 255])
        else:
            img = environment.set_tile_color(img, x, y, [255, 0, 0])
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

    # Plot value function representation
    environment_bg_image = environment.get_environment_background(ignore_agent=True)
    oracle = environment.get_oracle()
    # Build goal that will condition the value function
    # (for any observation, we shot how interesting it is to reach this goal).
    goal = environment.get_state(6, 13)
    # -> To change value, coordinates belong to the one on the map located at src/environments/grid_world/maps/map6.txt

    values = []
    for state in oracle:
        # Get value
        goal_conditioned_state = torch.from_numpy(np.concatenate((state, goal)))
        q_value = max(agent.target_model(goal_conditioned_state)).item()
        values.append(q_value)
    # Get value color
    mini = min(values)
    maxi = max(values)
    diff = maxi - mini
    for state, value in zip(oracle, values):
        x, y = environment.get_coordinates(state)
        normalised_value = (value - mini) / diff
        assert 0 <= normalised_value <= 1
        color = get_red_green_color(normalised_value, hexadecimal=False)
        environment_bg_image = environment.set_tile_color(environment_bg_image, x, y, color)

    # Put a black tile at the goal location
    x, y = environment.get_coordinates(goal)
    environment_bg_image = environment.set_tile_color(environment_bg_image, x, y, [0, 0, 0])
    ax_value_function_representation.imshow(environment_bg_image)  # Plot image

    ax_value_function_representation.set_title("Mini (red) = " + str(mini) + "; maxi (green) = " + str(maxi))

    ax_train_goals.set_title("Goals sampled for training from agent's memory")
    ax_train_accuracy.set_title("Accuracy of agent during training")
    ax_train_accuracy.set_xlabel("episodes")
    ax_train_accuracy.set_ylabel("accuracy")

    plt.show()
    plt.pause(.001)

    if save_path is not None:
        plt.savefig(save_path)


# Prepare plots
_, axs = init_plots(nb_rows=2, nb_cols=2)
ax_train_goals, ax_train_accuracy, ax_value_function_representation, _ = axs.flat

environment = GoalConditionedDiscreteGridWorld(map_id=6)
target_goal = (6, 13)

dqn_seeds_rewards_result = []
dqn_seeds_tests_result = []
colors = ["#ff0000", "#ff9500"]

agent = DQNHERAgent(state_space=environment.observation_space, action_space=environment.action_space)

for seed_id in range(settings.nb_seeds):
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

update_plots(save_path="outputs/value_function_B.png")
time.sleep(15)
