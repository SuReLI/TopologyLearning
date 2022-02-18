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

for simulation in simulations:
    for seed_id in range(settings.nb_seeds):
        agent = simulation.agent
        assert isinstance(agent, TopologyLearner)
        agent.on_simulation_start()
        if settings.pre_train_low_level_agent:
            res = pre_train(agent.current_agent(), environment)
        done = False
        evaluation_id = 0

        # Initialise plots
        figure, subplots = initiate_plots(agent)

        # Train
        for episode_id in range(settings.nb_episodes_max):

            # Prepare video rendering
            video_output = False
            if settings.rendering_start_at_episode <= episode_id <= settings.rendering_stop_at_episode and \
                    settings.nb_episodes_between_two_records is not None and \
                    episode_id % settings.nb_episodes_between_two_records == 0:
                video_output = True
                episode_images = []

            # Start episode loop
            state = environment.reset()
            agent.on_episode_start(state, TopologyLearnerMode.LEARN_ENV, episode_id)
            if settings.nb_episode_before_graph_update is not None and \
                    (episode_id + 1) % settings.nb_episode_before_graph_update == 0:
                update_plots(figure, subplots, environment, simulation, simulations, episode_id)

            time_steps = 0
            while not done:
                # Record video ...
                if video_output:
                    image = environment.render('rgb_array')
                    episode_images.append(image)

                # MDP loop ...
                action = agent.action(state)
                state, _, _, _ = environment.step(action)
                agent.on_action_stop(action, state, None, None, train_policy=False)
                done = agent.done

                if done:
                    environment.close()
                    agent.on_episode_stop()

                    # Make tests if needed
                    if (episode_id + 1) % settings.nb_episodes_before_evaluation == 0:
                        reached_goals, failed_goals = evaluation(simulation, environment, evaluation_id)
                        evaluation_id += 1

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

                time_steps += 1
            done = False

        if settings.nb_episode_before_graph_update is not None:
            # Final simulation plot
            update_plots(figure, subplots, environment, simulation, simulations, 0)

        # Stop simulation ...
        agent.on_simulation_stop()
        simulation.data_holder.on_seed_end()

        plt.ioff()
        environment.close()

