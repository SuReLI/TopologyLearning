import os
import sys
from datetime import datetime

from pre_train import pre_train
from settings import settings
from settings.simulations import simulations, environment
from tests import evaluation
from utils.sys_fun import create_dir
from rendering import initiate_plots, update_plots
import matplotlib.pyplot as plt
from agents import TopologyLearnerMode, TopologyLearner

# Simulations starting, build the global output directory
create_dir(settings.global_output_directory)

# Create and set up a standard output file
if settings.redirect_std_output:
    standard_output_file_name = settings.global_output_directory + "std_output.txt"
    os.mknod(standard_output_file_name)
    print("standard output file created at " + standard_output_file_name)
    sys.stdout = open(standard_output_file_name, "w")

# Run simulations
for simulation in simulations:
    for seed_id in range(settings.nb_seeds):
        # Change the seed output directory to the new seed
        simulation.outputs_directory = simulation.outputs_directory[:-2] + str(seed_id) + "/"

        # Set the simulation time counter (used to plot training time with graphs)
        simulation.start_time = datetime.now()
        simulation.pause_total_duration = simulation.start_time - simulation.start_time  # Time = 0 seconds

        # Set up  and pre-train agent if needed
        agent = simulation.agent
        assert isinstance(agent, TopologyLearner)
        agent.on_simulation_start()
        if settings.pre_train_low_level_agent:
            res = pre_train(agent.current_agent(), environment)
        done = False

        # Initialise plots
        figure, subplots = initiate_plots(agent)

        # Train
        interaction_id = 0
        episode_id = 0
        evaluation_id = 0  # The training loop will break one the number of evaluations is higher than the max allowed.
        while True:
            # Prepare video rendering
            video_output = False
            if episode_id != 0 and settings.rendering_start_at_episode <= episode_id and \
                    settings.nb_episodes_between_two_records is not None and \
                    episode_id % settings.nb_episodes_between_two_records == 0:
                video_output = True
                episode_images = []

            # Start episode loop
            if episode_id != 0 and settings.nb_episode_before_graph_update is not None and \
                    (episode_id + 1) % settings.nb_episode_before_graph_update == 0:
                update_plots(figure, subplots, environment, simulation, simulations, episode_id)
            state = environment.reset()
            agent.on_episode_start(state, TopologyLearnerMode.LEARN_ENV)

            time_steps = 0
            while not done:
                # Record video ...
                if video_output:
                    image = environment.render('rgb_array')
                    episode_images.append(image)

                # MDP loop ...
                action = agent.action(state)
                state, _, _, _ = environment.step(action)
                interaction_id += 1
                agent.on_action_stop(action, state, None, None, train_policy=False)
                done = agent.done

                # Evaluation if needed
                if interaction_id % settings.nb_interactions_before_evaluation == 0:
                    reached_goals, failed_goals = evaluation(simulation, evaluation_id)
                    evaluation_id += 1

                    evaluation_start_time = datetime.now()
                    _, _, _, _, _, ax6 = subplots.flat
                    ax6.clear()
                    img = environment.get_environment_background(ignore_agent=True)
                    for goal in reached_goals:
                        x, y = environment.get_coordinates(goal)
                        img = environment.set_tile_color(img, x, y, [0, 0, 255])
                    for goal in failed_goals:
                        x, y = environment.get_coordinates(goal)
                        img = environment.set_tile_color(img, x, y, [255, 0, 0])
                    ax6.imshow(img)
                    simulation.pause_total_duration += (datetime.now() - evaluation_start_time)
                    if evaluation_id >= settings.nb_evaluations_max:
                        break  # End the training loop

                if done:
                    environment.close()
                    agent.on_episode_stop()
                time_steps += 1
            done = False
            episode_id += 1
            if evaluation_id >= settings.nb_evaluations_max:
                break  # End the training loop

        if settings.nb_episode_before_graph_update is not None:
            # Final simulation plot
            update_plots(figure, subplots, environment, simulation, simulations, 0)

        # Stop simulation ...
        agent.on_simulation_stop()
        simulation.data_holder.on_seed_end()

        plt.ioff()
        environment.close()

        simulation.end_time = datetime.now()

