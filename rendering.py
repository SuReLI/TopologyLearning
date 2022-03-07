import copy
from datetime import datetime
from statistics import mean
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cv2

from settings import settings
from settings.simulations import Simulation
from agents import TopologyLearner
from utils.sys_fun import create_dir, get_red_green_color
import agents.graph_building_strategies as gbs


def initiate_plots(agent):
    plt.close(plt.figure(1))
    plt.ion()

    """
    figure = plt.figure(constrained_layout=True, figsize=(15, 10))
    sub_figures = figure.subfigures(1, 2)  # Build layout
    main_subplots = sub_figures.flat[0].subplots(2, 2)  # Initialize main subplots
    agent.initiate_subplots(sub_figures.flat[1])  # Initialize agent subplots
    """

    figure = plt.figure(constrained_layout=True, figsize=(15, 10))
    both = settings.plot_main_side and settings.plot_agent_side
    if both:
        sub_figures = figure.subfigures(1, 2)  # Build layout
    else:
        sub_figures = figure.subfigures(1, 1)  # Build layout
    main_subplots = None
    if settings.plot_main_side:
        sub_figure = sub_figures.flat[0] if both else sub_figures
        main_subplots = sub_figure.subplots(*settings.plot_main_side_shape)  # Initialize main subplots
    if settings.plot_agent_side:
        sub_figure = sub_figures.flat[1] if both else sub_figures
        agent.initiate_subplots(sub_figure)  # Initialize agent subplots

    return figure, main_subplots  # For main plots initialisation


def update_plots(figure, main_subplots, environment, current_simulation: Simulation, simulations: list,
                 episode_id: int):

    environment_image = environment.render()
    train_duration = (datetime.now() - current_simulation.start_time) - current_simulation.pause_total_duration
    figure.suptitle("Training duration: " + str(train_duration).split(".")[0], fontsize=16)

    # plot update
    current_simulation.agent.update_plots(environment)
    # Plot the oracle in blue and the projection in red
    ax1, ax2, ax3, ax4, ax5, ax6 = main_subplots.flat
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    """
    for ax in figure.axes:
        ax.clear()
    """
    assert isinstance(current_simulation.agent, TopologyLearner)

    if settings.plot_main_side:
        ax1.set_title("Reaching agent accuracy")
        if len(current_simulation.agent.reaching_success_moving_average) > 300:
            ax1.plot(current_simulation.agent.reaching_success_moving_average[-300:])
        else:
            ax1.plot(current_simulation.agent.reaching_success_moving_average)

        ax2.set_title("Nodes domination over environment")
        topology = copy.deepcopy(current_simulation.agent.topology)
        oracle = environment.get_oracle()
        colors = [[int(hexacol[i:i + 2], 16) for i in (1, 3, 5)] for hexacol in settings.colors]
        graph_coloration = nx.greedy_color(topology)
        background = environment.get_environment_background()
        for state in oracle:
            node_id = current_simulation.agent.get_node_for_state(state)
            state_color = colors[graph_coloration[node_id]]
            x, y = environment.get_coordinates(state)
            background = environment.set_tile_color(background, x, y, state_color)
        ax2.imshow(background)

        ax3.set_title("Environment topology")
        ax3.imshow(environment_image)

        """
        Plot the graph over our environment representation
        """

        # Compute environment image dimensions to get a weights scale on the image.
        # Note that nodes weights are between 0 and 1.
        topology = copy.deepcopy(current_simulation.agent.topology)
        image_height, image_width, _ = environment_image.shape
        nodes_coordinates = nx.get_node_attributes(topology, 'weights')
        scale = np.array([image_height, image_width])
        for node_id, coordinates in nodes_coordinates.items():
            nodes_coordinates[node_id] *= scale

        # Compute nodes colors
        colors_map = []
        risk_values = []

        for _, node_params in topology.nodes(data=True):
            risk_values.append(node_params["density"])

        high = max(risk_values)
        low = min(risk_values)
        distance = high - low
        if distance == 0:
            distance = 1e-6
        for elt, (node_id, node_params) in zip(risk_values, topology.nodes(data=True)):
            color = get_red_green_color((elt - low) / distance)
            if node_id == current_simulation.agent.last_node_explored:
                color = "#f4fc03"
            colors_map.append(color)

        # Compute edges color
        edges_colors = []
        if topology.edges():
            if isinstance(current_simulation.agent.topology_manager, gbs.TCNG):
                edges_colors = ["#000000" if params["strong"] else "#00ff00"
                                for _, _, params in topology.edges(data=True)]
            else:
                risk_values = [0 if params["risk"] == 1. else 1 for _, _, params in topology.edges(data=True)]
                max_risk = max(risk_values)
                min_risk = min(risk_values)
                diff_risk = max_risk - min_risk
                if diff_risk == 0:
                    edges_color_values = [1. for _ in risk_values]
                else:
                    edges_color_values = [1 - (risk_value - min_risk) / diff_risk for risk_value in risk_values]
                edges_colors = [get_red_green_color(color_value) for color_value in edges_color_values]

                for edge_index, (first, second) in enumerate(topology.edges(data=False)):
                    if (first, second) == current_simulation.agent.last_edge_failed \
                            or (second, first) == current_simulation.agent.last_edge_failed:
                        edges_colors[edge_index] = settings.failed_edge_color

        # Plot graph
        nx.draw(topology, nodes_coordinates, with_labels=False, node_color=colors_map, ax=ax3,
                edge_color=edges_colors, alpha=settings.nodes_alpha)
        nx.draw(topology, nodes_coordinates, with_labels=False, node_color=colors_map, ax=ax2,
                edge_color=edges_colors, alpha=settings.nodes_alpha)

        for node in topology:
            nx.draw_networkx_labels(topology, nodes_coordinates, labels={node: node}, font_color=settings.labels_color,
                                    ax=ax3)

        ax4.set_title("Goal distance to the closest node")
        ax5.set_title("Goal reaching accuracy")
        for simulation in simulations:
            # Plot the closest node distance from goal
            means, stds = simulation.data_holder.get_node_distances_evolution()
            abscissa_values = [x * settings.nb_interactions_before_evaluation for x in range(1, len(means) + 1)]
            ax4.plot(abscissa_values, means, color=simulation.color, label=simulation.agent.name)
            ax4.fill_between(abscissa_values, means + stds, means - stds,
                             color=simulation.color, alpha=settings.std_area_transparency)

            # Plot the average accuracy
            means, stds = simulation.data_holder.get_accuracy_evolution()
            ax5.plot(abscissa_values, means, color=simulation.color, label=simulation.agent.name)
            ax5.fill_between(abscissa_values, means + stds, means - stds,
                             color=simulation.color, alpha=settings.std_area_transparency)
        ax4.legend()
        ax5.legend()

    # Save the current image
    directory = current_simulation.outputs_directory + "graphs/"
    plot_filename = "plot_episode_" + str(episode_id)
    create_dir(directory)
    figure.savefig(directory + plot_filename)
    plt.pause(0.00001)
