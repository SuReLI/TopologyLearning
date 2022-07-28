"""
A script to test goal based RL agent, that are used to reach sub-goals.
"""
import copy
from copy import deepcopy
import time
from statistics import mean

import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gym.spaces import Box

from src.agents import SACHERAgent, SACHERDiffAgent
from src.agents.grid_world.graph_planning.sorb_no_oracle import SORB_NO_ORACLE
from src.agents.grid_world.graph_planning.sorb_oracle import SORB
from src.agents.grid_world.graph_planning.stc import STC_TL
from src.agents.grid_world.graph_planning.tipp import TIPP
from src.agents.grid_world.graph_planning.tipp_gwr import TIPP_GWR
from src.agents.grid_world.graph_planning.topological_graph_planning_agent import TopologyLearnerMode, \
    PlanningTopologyLearner
from src.settings import settings
from src.utils.plots import init_plots
import matplotlib.cm as cm
import d4rl
# Initialise data memories
from src.utils.pre_train_mujoco_maze import pre_train
from src.utils.sys_fun import get_red_green_color, create_dir, save_image

current_seed_test_accuracy_memory = []
current_seed_nb_nodes_memory = []
current_seed_color = "#000000"
tests_output_directory = "outputs/tests_outputs/"
create_dir(tests_output_directory)
eval_id = 0

PRE_TRAIN_DELTA = []
GRAPH_DELTA = []
GO_TO_DELTA = []
plot_deltas = False

# env_type = "pointmaze"
env_type = "antmaze"

if env_type == "pointmaze":
    state_to_goal_filter = np.array([1, 1, 0, 0])
else:
    state_to_goal_filter = np.zeros(29)
    state_to_goal_filter[:2] = np.ones(2)
state_to_goal_filter = state_to_goal_filter.astype(bool)


def reached(state, goal):
    global env_type
    if env_type == "pointmaze":
        tolerance_threshold = settings.point_maze_pre_train_reachability_threshold
    else:
        tolerance_threshold = settings.ant_maze_pre_train_reachability_threshold
    s = state[state_to_goal_filter]
    diff = s - goal
    return np.linalg.norm(diff, 2) < tolerance_threshold
    # and state[-1] - goal[-1] < settings.point_maze_pre_train_velocity_threshold


def sample_pre_train_target_state(environment, state_to_goal_filter: np.ndarray) -> np.ndarray:
    """
    This function is a default version. The used one will be given to the pre_train method.
    Sample a state that can be targeted during pre_training.

    :parameter environment: Environment the agent is inside.
    :parameter state_to_goal_filter: numpy array of booleans, that can be used as a filter on a state to convert it
    into a goal.
    """
    global env_type
    if env_type == "pointmaze":
        # Sample location
        goal = environment.sample_goal()

        # Sample target velocity
        velocity_range = 3
        target_velocity = np.random.rand(2) * (velocity_range * 2) - velocity_range
        return np.concatenate((goal, target_velocity))[state_to_goal_filter]
    else:
        assert len(environment.maze_map[0]) == 5, len(environment.maze_map) == 5
        mini = environment.tile_bounds[0] + 1
        maxi = environment.tile_bounds[1] + 3
        goal_shape = (2,)
        row_col_goal = Box(np.full(goal_shape, mini), np.full(goal_shape, maxi)).sample()
        xy_goal = environment.rowcol_to_xy(row_col_goal)
        goal = np.array(xy_goal)
        return goal


def run_simulation(_simulation):
    _agent = _simulation["agent"]
    global environment, PRE_TRAIN_DELTA, GRAPH_DELTA, GO_TO_DELTA, plot_deltas, env_type

    _agent.on_simulation_start()
    if isinstance(_agent, PlanningTopologyLearner) or isinstance(_agent, SORB):
        if env_type == "pointmaze":
            pre_train_sand_box_map_spec = \
                "#####\\" + \
                "#OOO#\\" + \
                "#OSO#\\" + \
                "#OOO#\\" + \
                "#####"
            pre_train_environment = gym.make("maze2d-umaze-v1", maze_spec=pre_train_sand_box_map_spec)
        else:
            pre_train_sand_box_map_spec = [[1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 0, 'r', 0, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]]
            pre_train_environment = gym.make("antmaze-umaze-v2", maze_map=pre_train_sand_box_map_spec)
        PRE_TRAIN_DELTA = pre_train(_agent, pre_train_environment, state_to_goal_filter=state_to_goal_filter,
                                    goal_sampler=sample_pre_train_target_state, reach_criteria=reached)

    # Train
    interaction_id = 0
    evaluation_id = 0
    episode_id = 0
    if env_type == "point_maze":
        nb_evaluations = settings.point_maze_nb_evaluations_max
    else:
        nb_evaluations = settings.ant_maze_nb_evaluations_max
    while evaluation_id < nb_evaluations:
        state, goal = environment.reset()
        if isinstance(_agent, PlanningTopologyLearner):
            _agent.on_episode_start(state, TopologyLearnerMode.LEARN_ENV)
        elif isinstance(_agent, SORB):
            _agent.on_episode_start(state, goal)
        else:
            _agent.on_episode_start(state, None)

        if plot_deltas:
            current_pos = deepcopy(state)
            for node_id in _agent.current_exploration_nodes_path:
                next_goal = _agent.topology.nodes[node_id]["state"]
                delta = current_pos - next_goal
                current_pos = deepcopy(next_goal)

                for d in GRAPH_DELTA:
                    if (d == delta).all():
                        break
                else:
                    GRAPH_DELTA.append(delta)

            delta = current_pos - np.concatenate((goal, np.zeros(2)))
            for d in GO_TO_DELTA:
                if (d == delta).all():
                    break
            else:
                GO_TO_DELTA.append(delta)

        while not _agent.done:
            action = _agent.action(state)
            state, _, _, _ = environment.step(action)
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
    PRE_TRAIN_DELTA = []
    GRAPH_DELTA = []
    GO_TO_DELTA = []


def evaluation(current_simulation_agent):
    global environment_name
    global ax_test_goals
    global test_id, eval_id

    test_environment = gym.make(environment_name)
    #  '-> So we can test our agent at any time in a parallel environment, even in the middle of an episode

    # Get an agent copy and prepare it to the test
    test_agent = copy.deepcopy(current_simulation_agent)
    if isinstance(test_agent, PlanningTopologyLearner):
        test_agent.on_episode_stop(learn=False)
    else:
        test_agent.on_episode_stop()
    results = []
    for test_id in range(settings.nb_tests):
        goal, result = test(test_agent, test_environment)
        results.append(result)
    current_seed_test_accuracy_memory.append(mean(results))
    if hasattr(test_agent, "topology"):
        current_seed_nb_nodes_memory.append(len(test_agent.topology.nodes))
    test_id = 0
    eval_id += 1


def test(_agent, _environment):
    """
    Test the agent over a single goal reaching task. Return the result that will be directly passed to the DataHolder.
    return tuple(the closest node distance from goal, success in {0, 1})
    """
    global eval_id, test_id, env_type
    directory = tests_output_directory + "eval_" + str(eval_id) + "/test_" + str(test_id) + "/"
    test_id += 1
    create_dir(directory)

    state, goal = _environment.reset()
    if isinstance(_agent, PlanningTopologyLearner):
        _agent.on_episode_start(state, TopologyLearnerMode.GO_TO, goal)  # reset our agent copy
    else:
        _agent.on_episode_start(state, goal)

    reached = False
    image_id = 0
    while not reached or not _agent.done:
        image = _environment.get_background_image()
        image = _environment.state_on_image(_agent.next_goal, image=image)
        image = _environment.state_on_image(goal, image=image, color=[0, 255, 0])
        image = _environment.state_on_image(state, image=image, color=[0, 0, 255])
        image_name = str(image_id) + ".png"
        image_id += 1
        save_image(image, directory, image_name)

        action = _agent.action(state)
        state, _, _, _ = _environment.step(action)
        if env_type == "pointmaze":
            tolerance_threshold = settings.point_maze_nodes_reachability_threshold
        else:
            tolerance_threshold = settings.ant_maze_nodes_reachability_threshold
        if np.linalg.norm(state[:2] - goal, 2) <= tolerance_threshold:
            reached = True
        _agent.on_action_stop(action, state, None, None, learn=False)
        if reached or _agent.done:
            _agent.on_episode_stop()
            return goal, 1 if reached else 0
    raise Exception("Maximum time steps reached for a test")


def update_plots(current_simulation, save_path=None):
    global PRE_TRAIN_DELTA, GRAPH_DELTA, GO_TO_DELTA, plot_deltas
    global simulations
    global environment
    global ax_graph_representation, ax_test_accuracy
    global current_seed_test_accuracy_memory
    print("Updating_plots")

    # Update test accuracy graph
    ax_test_accuracy.set_title("Agent's accuracy over tests")
    ax_test_accuracy.set_xlabel("episodes")
    ax_test_accuracy.set_ylabel("accuracy")
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
            ax_test_accuracy.fill_between(abscissa_values, means + stds, means - stds,
                                          alpha=settings.std_area_transparency, color=simulation["color"])
    if current_seed_test_accuracy_memory:
        legend = True
        abscissa_values = \
            [x * settings.nb_episodes_before_tests for x in range(1, len(current_seed_test_accuracy_memory) + 1)]
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

    if isinstance(current_simulation["agent"], PlanningTopologyLearner) or \
            isinstance(current_simulation["agent"], SORB):

        if plot_deltas:
            ax.cla()
            pre_train_data = np.array(PRE_TRAIN_DELTA)
            graph_data = np.array(GRAPH_DELTA)
            go_to_data = np.array(GO_TO_DELTA)

            data_min = min(np.concatenate((pre_train_data[:, 2], graph_data[:, 2], go_to_data[:, 2])))
            data_max = max(np.concatenate((pre_train_data[:, 2], graph_data[:, 2], go_to_data[:, 2])))

            col_values = (pre_train_data[:, 2] - data_min) / (data_max - data_min)
            colors = np.vectorize(get_red_green_color)(col_values)
            collection = ax.scatter(pre_train_data[:, 0], pre_train_data[:, 1], pre_train_data[:, 3],
                                    c=colors, vmin=data_min, vmax=data_max, marker='o', cmap=cm.Spectral,
                                    label="pre train deltas")

            col_values = (graph_data[:, 2] - data_min) / (data_max - data_min)
            colors = np.vectorize(get_red_green_color)(col_values)
            collection = ax.scatter(graph_data[:, 0], graph_data[:, 1], graph_data[:, 3], c=colors, vmin=data_min,
                                    vmax=data_max, marker='^', cmap=cm.Spectral, label="deltas in graph")

            col_values = (go_to_data[:, 2] - data_min) / (data_max - data_min)
            colors = np.vectorize(get_red_green_color)(col_values)
            collection = ax.scatter(go_to_data[:, 0], go_to_data[:, 1], go_to_data[:, 3], c=colors, vmin=data_min,
                                    vmax=data_max, marker='s', cmap=cm.Spectral, label="deltas in final_go_to")

            ax.set_xlabel('x pos')
            ax.set_ylabel('y pos')
            ax.set_zlabel('velocity')
            ax.legend()

        # Plot agent's topology
        ax_graph_representation.clear()
        ax_graph_representation.set_title("Topological graph")

        # Plot the graph over our environment representation
        environment_background_image = environment.get_background_image()
        ax_graph_representation.imshow(environment_background_image)

        # Compute environment image dimensions to get a weights scale on the image.
        # Note that nodes weights are between 0 and 1.
        topology = copy.deepcopy(current_simulation["agent"].topology)
        nodes_coordinates = nx.get_node_attributes(topology, 'state')
        image_height, image_width, _ = environment_background_image.shape
        scale = np.array([image_height, image_width])
        for node_id, coordinates in nodes_coordinates.items():
            nodes_coordinates[node_id] = environment.get_normalised_position(np.flip(coordinates[:2]),
                                                                             scaled_on_image=True) * scale
        GO_TO_DELTA = []

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
                    label_dict[node] = node

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
                edge_color=edges_colors, alpha=settings.nodes_alpha, labels=label_dict, node_size=settings.nodes_size)

        ax_nb_nodes.cla()
        if current_simulation["agent"].s_g_s:
            data = np.array(current_simulation["agent"].s_g_s)
            ax_nb_nodes.scatter(data[:, 0], data[:, 1])
        if current_simulation["agent"].s_g_f:
            data = np.array(current_simulation["agent"].s_g_f)
            ax_nb_nodes.scatter(data[:, 0], data[:, 1], c="r")

    plt.show()
    plt.pause(.0001)

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

if plot_deltas:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

if env_type == "pointmaze":
    # environment_name = "maze2d-umaze-v1"
    environment_name = "maze2d-large-v1"
    # environment_name = "maze2d-huge-v1"
    environment = gym.make(environment_name)
else:
    environment_name = "antmaze-umaze-v2"
    environment = gym.make(environment_name)

"""
HER
HER Diff (s - g)
"""
# DDPGHERDiffAgent
# SACHERDiffAgent2

if env_type == "pointmase":
    tolerance_radius = settings.point_maze_nodes_reachability_threshold
else:
    tolerance_radius = settings.ant_maze_nodes_reachability_threshold
simulations = [
    {
        "agent": TIPP(state_space=environment.observation_space, action_space=environment.action_space,
                      tolerance_radius=tolerance_radius,
                      goal_reaching_agent_class=SACHERDiffAgent, verbose=False,
                      edges_similarity_threshold=0.15, nodes_similarity_threshold=0.25,
                      max_steps_to_reach=70, state_to_goal_filter=state_to_goal_filter)
    },
    {
        "agent": STC_TL(state_space=environment.observation_space, action_space=environment.action_space,
                        tolerance_radius=tolerance_radius,
                        goal_reaching_agent_class=SACHERAgent),
        "color": "#ff0000"
    },
    {
        "agent": SACHERAgent(state_space=environment.observation_space, action_space=environment.action_space),
        "color": "#0000ff"
    },
    {
        "agent": SORB_NO_ORACLE(state_space=environment.observation_space, action_space=environment.action_space,
                                tolerance_radius=tolerance_radius,
                                goal_reaching_agent_class=SACHERAgent)
    },
    {
        "agent": TIPP_GWR(state_space=environment.observation_space, action_space=environment.action_space,
                          tolerance_radius=tolerance_radius,
                          goal_reaching_agent_class=SACHERDiffAgent)
    }
]

# Give a specific color to each simulation
assert len(simulations) <= len(settings.colors), "Too many simulations, add more colors to settings to run it."
for simulation_id, simulation in enumerate(simulations):
    simulation["color"] = settings.colors[simulation_id]
    simulation["last_seeds_test_accuracy_memories"] = []
    simulation["last_seeds_nb_nodes_memories"] = []

for simulation in simulations:
    print("goal reaching agent class is: ", simulation["agent"].goal_reaching_agent_class)
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
