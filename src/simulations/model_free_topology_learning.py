"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import copy
import time
from random import choice

import networkx as nx
import numpy as np

from src.agents import AutonomousDQNHERAgent
from src.agents.grid_world.graph_planning.sorb_no_oracle import SORB_NO_ORACLE
from src.agents.grid_world.graph_planning.sorb_oracle import SORB
from src.agents.grid_world.graph_planning.tipp import TIPP
from src.agents.grid_world.graph_planning.stc import STC_TL
from src.agents.grid_world.graph_planning.topological_graph_planning_agent import TopologyLearnerMode, \
    PlanningTopologyLearner
from src.environments import GoalConditionedDiscreteGridWorld
from src.settings import settings
from statistics import mean
import matplotlib.pyplot as plt
from src.utils.plots import init_plots
from src.agents.grid_world.graph_planning.tipp_gwr import TIPP_GWR

# Initialise data memories
from src.utils.pre_train import pre_train
from src.utils.sys_fun import get_red_green_color, create_dir

current_seed_test_accuracy_memory = []
current_seed_nb_nodes_memory = []
current_seed_color = "#000000"


def run_simulation(_simulation):
    _agent = _simulation["agent"]
    global environment

    _agent.on_simulation_start()
    if isinstance(_agent, PlanningTopologyLearner) or isinstance(_agent, SORB):
        pre_train(_agent, environment)

    # Train
    interaction_id = 0
    evaluation_id = 0
    episode_id = 0
    while evaluation_id < settings.nb_evaluations_max:
        state, goal = environment.reset()
        if isinstance(_agent, PlanningTopologyLearner):
            _agent.on_episode_start(state, TopologyLearnerMode.LEARN_ENV)
        elif isinstance(_agent, SORB):
            _agent.on_episode_start(state, goal)
        else:
            _agent.on_episode_start(state, None)

        while not _agent.done:
            action = _agent.action(state)
            state, _, _ = environment.step(action)
            interaction_id += 1
            _agent.on_action_stop(action, state, None, None)

            # Evaluation if needed
            if interaction_id != 0 and interaction_id % settings.nb_interactions_before_evaluation == 0:
                # evaluation_start_time = datetime.now()
                evaluation(_agent)
                evaluation_id += 1
                # pause_total_duration += (datetime.now() - evaluation_start_time)
        _agent.on_episode_stop()

        episode_id += 1
        if episode_id % settings.nb_episodes_before_plots == 0 and settings.nb_episodes_before_plots is not None:
            # Final simulation plot
            update_plots(_simulation)

    # Stop simulation ...
    _agent.on_simulation_stop()
    # end_time = datetime.now()


def evaluation(current_simulation_agent):
    global environment
    global ax_test_goals
    env = copy.deepcopy(environment)
    #  '-> So we can test our agent at any time in a parallel environment, even in the middle of an episode

    ax_test_goals.clear()
    environment_bg_image = env.get_environment_background(ignore_agent=True)

    # Get an agent copy and prepare it to the test
    test_agent = copy.deepcopy(current_simulation_agent)
    if isinstance(test_agent, PlanningTopologyLearner):
        test_agent.on_episode_stop(learn=False)
    else:
        test_agent.on_episode_stop()
    results = []
    for test_id in range(settings.nb_tests):
        goal, result = test(test_agent, env)
        results.append(result)
        color = [0, 0, 255] if result else [255, 0, 0]
        environment_bg_image = env.set_tile_color(environment_bg_image, *environment.get_coordinates(goal), color)
    current_seed_test_accuracy_memory.append(mean(results))
    if hasattr(test_agent, "topology"):
        current_seed_nb_nodes_memory.append(len(test_agent.topology.nodes))
    ax_test_goals.imshow(environment_bg_image)  # Plot image
    ax_test_goals.set_title("Goals sampled from oracle on tests")
    plt.show()
    plt.pause(.001)


def test(_agent, _environment):
    """
    Test the agent over a single goal reaching task. Return the result that will be directly passed to the DataHolder.
    return tuple(the closest node distance from goal, success in {0, 1})
    """
    state, goal = _environment.reset()                                        # Reset our environment copy
    if isinstance(_agent, PlanningTopologyLearner):
        _agent.on_episode_start(state, TopologyLearnerMode.GO_TO, goal)     # reset our agent copy
    else:
        _agent.on_episode_start(state, goal)

    reached = False
    test_duration = 0
    while not reached or not _agent.done:
        action = _agent.action(state)
        state, _, _ = _environment.learning_step(action)
        _agent.on_action_stop(action, state, None, None, learn=False)
        reached = (state == goal).all()
        test_duration += 1
        if reached or _agent.done:
            _agent.on_episode_stop()
            return goal, 1 if reached else 0
    raise Exception("Maximum time steps reached for a test")


def update_plots(current_simulation, save_path=None):
    global simulations
    global environment
    global ax_graph_representation, ax_test_accuracy
    global last_seeds_test_accuracy_memories, current_seed_test_accuracy_memory, tc_similarity_estimation_plot

    # Update others plots
    # Update sampled goals representation

    # Update test accuracy graph
    ax_test_accuracy.clear()
    legend = False
    for simulation in simulations:
        if simulation["last_seeds_test_accuracy_memories"]:
            legend = True
            data = np.array(simulation["last_seeds_test_accuracy_memories"])
            means = np.mean(data, 0)
            stds = np.std(data, 0) if len(data > 0) else 0
            abscissa_values = [x * settings.nb_episodes_before_tests for x in range(1, len(means) + 1)]
            ax_test_accuracy.plot(abscissa_values, means, label=simulation["agent"].name, color=simulation["color"])
            ax_test_accuracy.fill_between(abscissa_values, means + stds, means - stds, alpha=settings.std_area_transparency,
                                          color=simulation["color"])
    if current_seed_test_accuracy_memory:
        legend = True
        abscissa_values = [x * settings.nb_episodes_before_tests for x in range(1, len(current_seed_test_accuracy_memory) + 1)]
        ax_test_accuracy.plot(abscissa_values, current_seed_test_accuracy_memory, "--",
                              label=current_simulation["agent"].name + " running seed",
                              color=current_simulation["color"])

    # Plot nb nodes in graph
    ax_nb_nodes.clear()
    for simulation in simulations:
        if simulation["last_seeds_nb_nodes_memories"]:
            data = np.array(simulation["last_seeds_nb_nodes_memories"])
            means = np.mean(data, 0)
            stds = np.std(data, 0) if len(data > 0) else 0
            abscissa_values = [x * settings.nb_episodes_before_tests for x in range(1, len(means) + 1)]
            ax_nb_nodes.plot(abscissa_values, means, label=simulation["agent"].name, color=simulation["color"])
            ax_nb_nodes.fill_between(abscissa_values, means + stds, means - stds, alpha=settings.std_area_transparency,
                                     color=simulation["color"])
    if current_seed_nb_nodes_memory:
        abscissa_values = [x * settings.nb_episodes_before_tests for x in range(1, len(current_seed_nb_nodes_memory) + 1)]
        ax_nb_nodes.plot(abscissa_values, current_seed_nb_nodes_memory, "--",
                         label=current_simulation["agent"].name + " running seed", color=current_simulation["color"])

    if legend:
        ax_test_accuracy.legend()
        ax_nb_nodes.legend()
    ax_nb_nodes.set_title("How many nodes in graph")
    ax_test_accuracy.set_title("Agent's accuracy over tests")
    ax_test_accuracy.set_xlabel("episodes")
    ax_test_accuracy.set_ylabel("accuracy")

    if isinstance(current_simulation["agent"], PlanningTopologyLearner) or \
            isinstance(current_simulation["agent"], SORB):

        # Plot agent's topology over the environment representation
        ax_graph_representation.clear()
        ax_graph_representation.set_title("Topological graph over environment representation.")

        environment_background_image = environment.get_environment_background(ignore_agent=True)
        ax_graph_representation.imshow(environment_background_image)

        # Plot the graph over our environment representation

        # Compute environment image dimensions to get a weights scale on the image.
        # Note that nodes weights are between 0 and 1.
        topology = copy.deepcopy(current_simulation["agent"].topology)
        image_height, image_width, _ = environment_background_image.shape
        nodes_coordinates = nx.get_node_attributes(topology, 'observation')
        scale = np.array([image_height, image_width])
        for node_id, coordinates in nodes_coordinates.items():
            nodes_coordinates[node_id] *= scale

        # Compute nodes colors
        colors_map = []
        label_dict = {}
        if topology.nodes():
            nodes_value = []
            if isinstance(current_simulation["agent"], SORB):
                for _ in topology.nodes():
                    colors_map.append("#00ff00")
            else:
                nodes_valuable_attribute = "explorations" if current_simulation["agent"].re_usable_policy else "reached"

                for node, node_params in topology.nodes(data=True):
                    nodes_value.append(node_params[nodes_valuable_attribute])
                    label_dict[node] = node_params[nodes_valuable_attribute]

                high = max(nodes_value)
                low = min(nodes_value)
                distance = high - low
                if distance == 0:
                    distance = 1e-6
                for elt, (node_id, node_params) in zip(nodes_value, topology.nodes(data=True)):
                    color = get_red_green_color((elt - low) / distance)
                    if node_id == current_simulation["agent"].last_node_explored:
                        color = "#f4fc03"
                    colors_map.append(color)

        # Compute edges color
        edges_colors = []
        if isinstance(current_simulation["agent"], SORB):
            for _ in topology.nodes():
                edges_colors.append("#00ff00")
        else:
            if topology.edges():
                for _, _, params in topology.edges(data=True):
                    if "potential" in params.keys() and params["potential"]:
                        color = "#ffa500"
                    elif "exploration_cost" in params.keys() and params["exploration_cost"] == 1.:
                        color = "#00ff00"
                    else:
                        color = "#ff0000"
                    edges_colors.append(color)

        # Plot graph
        nx.draw(topology, nodes_coordinates, with_labels=True, node_color=colors_map, ax=ax_graph_representation,
                edge_color=edges_colors, alpha=settings.nodes_alpha, labels=label_dict)

    plt.show()
    plt.pause(.001)

    if save_path is not None:
        plt.savefig(save_path)


def save_results(directory, filename, agent):

    f = open("demofile2.txt", "a")
    f.write("Now the file has more content!")
    f.close()


# Prepare plots
fig, axs = init_plots(nb_rows=2, nb_cols=2)
fig.suptitle('Learn a topology')
ax_graph_representation, ax_test_goals, ax_nb_nodes, ax_test_accuracy = axs.flat
plt.show()
plt.pause(.001)

environment = GoalConditionedDiscreteGridWorld(map_id=7, stochasticity=0.0)
tiles_dimensions = environment.get_tile_dimensions()
"""
HER
HER Diff (s - g)

"""

simulations = [
    {
        "agent": STC_TL(state_space=environment.observation_space, action_space=environment.action_space,
                        tolerance_margin=tiles_dimensions, random_exploration_duration=100),
        "color": "#ff0000"
    },
    {
        "agent": SORB(state_space=environment.observation_space, action_space=environment.action_space,
                      tolerance_margin=tiles_dimensions, random_exploration_duration=100,
                      oracle=environment.get_oracle())
    },
    {
        "agent": TIPP(state_space=environment.observation_space, action_space=environment.action_space,
                      tolerance_margin=tiles_dimensions, random_exploration_duration=100)
    },
    {
        "agent": AutonomousDQNHERAgent(state_space=environment.observation_space,
                                       action_space=environment.action_space),
        "color": "#0000ff"
    },
    {
        "agent": SORB_NO_ORACLE(state_space=environment.observation_space, action_space=environment.action_space,
                                tolerance_margin=tiles_dimensions, random_exploration_duration=100)
    },
    {
        "agent": TIPP_GWR(state_space=environment.observation_space, action_space=environment.action_space,
                          tolerance_margin=tiles_dimensions, random_exploration_duration=100)
    }
]


# Give a specific color to each simulation
assert len(simulations) <= len(settings.colors), "Too many simulations, add more colors to settings to run it."
for simulation_id, simulation in enumerate(simulations):
    simulation["color"] = settings.colors[simulation_id]
    simulation["last_seeds_test_accuracy_memories"] = []
    simulation["last_seeds_nb_nodes_memories"] = []

for simulation in simulations:
    current_seed_color = simulation["color"]
    print()
    print("###################")
    print()
    print("  > TRAINING AGENT " + str(simulation["agent"].name))
    print()
    print("###################")
    print()
    for seed_id in range(settings.nb_seeds):
        print()
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print()
        print("      SEED " + str(seed_id))
        print()
        ax_graph_representation.clear()
        ax_test_goals.clear()
        run_simulation(simulation)

        simulation["last_seeds_test_accuracy_memories"].append(current_seed_test_accuracy_memory)
        current_seed_test_accuracy_memory = []
        simulation["last_seeds_nb_nodes_memories"].append(current_seed_nb_nodes_memory)
        current_seed_nb_nodes_memory = []

        if seed_id < settings.nb_seeds - 1:
            # Reset everything for next seed
            simulation["agent"].reset()

            # Check build tolerance margin/radius
            # Check node 0 position

    save_directory = "outputs/grid_world/model_free_topology_learning/"
    create_dir(save_directory)
    update_plots(simulation, save_path=save_directory + simulation["agent"].name.replace(" + ", "_") + "_s0,0.png")
time.sleep(15)
plt.ioff()
