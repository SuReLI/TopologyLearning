import copy
import inspect
import math
import os.path
import pickle
from random import random
from typing import Union

import networkx as nx
import numpy as np
from gym.spaces import Discrete, Box
from networkx import NetworkXNoPath, NetworkXError
from agents.agent import Agent
from agents.value_based_agent import ValueBasedAgent
from agents.goal_conditioned_wrappers.tilo import TILO
from utils import create_dir


class RGL(Agent):
    """
    RGL stands for Reachability Graph Learning
    An agent that is trained to learn the environment reachability_graph, so that learn by interacting with its
    environment, but don't need to reach a goal to do so. Then, he is able to exploit his knowledge of his environment
    to reach goals or to patrol inside it.
    self.mode is used to make the agent know what he should do during this episode.
    """

    name = "RGL"

    # def __init__(self, goal_conditioned_wrapper, value_based_agent_class,
    #              state_space: Union[Box, Discrete], action_space: Union[Box, Discrete], environment=None, # TODO: ENV
    #              **params):
    def __init__(self, goal_conditioned_wrapper, value_based_agent_class,
                 state_space: Union[Box, Discrete], action_space: Union[Box, Discrete],
                 **params):
        """
        @param goal_conditioned_wrapper: Wrapper used to turn the given value based agent class into a goal
            conditioned agent.
        @param value_based_agent_class: Value based agent class.
        @param state_space: Environment's state space.
        @param action_space: Environment's action_space.
        @param params: Optional parameters.
        """

        super().__init__(state_space, action_space, **params)

        assert issubclass(goal_conditioned_wrapper, TILO)
        self.control_policy_goal_conditioned_wrapper = goal_conditioned_wrapper
        self.control_policy_value_based_agent_class = value_based_agent_class

        # ALGORITHM TYPE
        # re_usable_policy=True implies a pre-training to the given policy, and that this policy will not be trained
        # during learning..
        self.re_usable_policy = params.get("re_usable_policy", True)

        # Margin distance under which a goal can be considered reached
        # Ex.: s = (x, y), g = (x', y'), tolerance_margin = (a, b)
        # goal is reached by state s if x' - x <= a and y' - y <= b.
        self.tolerance_radius = params.get("tolerance_radius", .8)

        # GRAPH BUILDING HYPER-PARAMETERS
        self.nodes_attributes = params.get("nodes_attributes", {})
        self.edges_attributes = params.get("edges_attributes", {})
        #  '--> Default nodes and edges attributes on creation as a dict, like {nb_explorations: 0} for nodes.

        self.exploration_duration = params.get("exploration_duration", 90)
        #  '--> Duration of any explorations from a node we want to explore from, in number of interactions.

        # SUB-GOALS PLANNING ARGUMENTS
        self.max_steps_to_reach = params.get("max_steps_to_reach", 50)

        # MISC. ARGUMENTS
        self.verbose = params.get("verbose", False)
        # Copy params and store them in self
        self.init_params = {}
        for k, v in params.items():
            self.init_params[k] = copy.deepcopy(v) if k != "goal_reaching_agent" else v.copy()

        # LOW-LEVEL ACTIONS PLANNING ARGUMENTS

        """
        General attributes (for any mode)
        """
        self.directed_graph = params.get("directed_graph", True)
        self.reachability_graph = nx.DiGraph() if self.directed_graph else nx.Graph()
        self.learn = True
        self.default_state = params.get("default_state")
        assert self.default_state is not None

        # Compute out goal space
        self.goal_space = params.get("goal_space", self.state_space)
        assert isinstance(self.goal_space, Box) or isinstance(self.goal_space, Discrete)
        self.goal_size = self.goal_space.shape[0]
        self.goal_shape = self.goal_space.shape
        assert len(self.goal_shape) == 1, "Multi dimensional spaces are not supported."

        # Compute state_to_goal_filter, (state[self.state_to_goal_filter] = <goal associated to state>) aka a projection
        default_filter = np.array([1] * self.goal_size + [0] * (self.state_size - self.goal_size)).astype(bool)
        self.state_to_goal_filter = params.get("state_to_goal_filter", default_filter)
        assert np.argwhere(self.state_to_goal_filter).shape[0] == self.goal_size

        if inspect.isclass(value_based_agent_class) and issubclass(value_based_agent_class, ValueBasedAgent):
            self.control_policy: TILO = goal_conditioned_wrapper(value_based_agent_class, state_space=self.state_space,
                                                                 action_space=self.action_space, **params)
        else:
            self.control_policy = value_based_agent_class  # Specific case for ant maze policy

        # The number of episode will depend on the task we are doing. In any mode, the agent choose when he's done doing
        # the task.
        #   For exploration, our agent will continue its environment while he didn't finish his exploration strategy.
        #   For goal reaching mode, our agent will continue until he reached the goal or until he considers that he
        # failed to reach it.
        #   For patrol, our agent will fix a custom time limit.
        self.done = False

        # In any mode, we will try to follow trajectories inside the learned reachability graph. Doing so, if we
        # failed reaching the next node, the entire path is compromised. To make sure we don't try infinitely to reach
        # an unreachable sub-goal, we should count how many time steps we tried to reach it so far, to detect it as an
        # impossible subtask. This allows us to break and start a new episode instead of looping infinitely.
        self.current_subtask_steps = 0  # Steps done so far to reach the next node.
        self.last_node_passed = None  # So we can know which edge is concerned if it failed.
        self.next_way_point = None

        """
        EXPLORATION AND GRAPH BUILDING ATTRIBUTES
            Explorations steps are simple:
             - Select a node to explore from (the one less chosen for exploration)
             - Find a path to this node in the topological graph using A*, Dijkstra, or any path finding algorithm,
             - For each waypoint, try to reach it, until we don't.
             - Once we reached the last sub-goal (aka. The node initially selected for exploration), perform randoms
             actions for a fixed duration, to sample random states next to this node.  
             - Use the randomly sampled states to add new nodes to our graph.
        """
        # Once we reached a node that has been selected as interesting for exploration, we will explore using a random
        # policy for a fixed duration.
        self.global_path = None
        self.higher_node_id = -1  # Identifier of the younger node. To know which id to give to a new node.
        self.select_goal_for_exploration = params.get("select_goal_for_exploration", True)
        self.exploration_goal_range = params.get("exploration_goal_range", 2)
        # Trajectory made once we reached last exploration waypoint.
        self.last_exploration_trajectory = []
        self.distance_estimation_max = None
        self.edges_distance_threshold = params.get("edges_distance_threshold", 0.7)
        self.nodes_distance_threshold = params.get("nodes_distance_threshold", 0.5)
        self.sampled_exploration_goal = None
        self.under_exploration = False
        # We don't want to perform (state, goal) distance on states and goals that are too distant from each others,
        # otherwise, we can feed into the policy neural networks g - s data that are too different from the ones
        # seen in the pre-training, and get fake d_pi(g - P(s)) results. This is the maximum acceptable distance for it.
        self.s_g_max_dist = params.get("s_g_max_dist", 19)
        self.default_state = None

        """
        GOAL REACHING ATTRIBUTES
        """
        self.final_goal = None

        # TODO: remove
        self.exploration_node = None
        self.node_from = None
        # self.environment = environment  # TODO: ENV

    """
    AGENT LIFECYCLE FUNCTIONS
    """
    def on_pre_training_done(self, start_state, reached_goals):
        """
        Compute the longer distance estimation over every goal that has been reached during the pre-training.
        It allows to choose reachability parameters more easily.
        """
        self.distance_estimation_max = None
        start_state_batch = np.tile(start_state, (len(reached_goals), 1))
        goals_batch = np.array(reached_goals)
        distance_estimations = self.control_policy.get_estimated_distances(start_state_batch, goals_batch)
        self.distance_estimation_max = distance_estimations.max()

    def start_episode(self, state: np.ndarray, goal=None, test_episode=False, node_to_explore=None):
        """
        Initialise an episode using the given 'state' as the initial state, and the given goal as a goal to reach.
        If the goal argument is set, the agent will consider that he is under test, and will not learn anything.
            (fail to cross an edge will not lead to modify its weight). Otherwise, the agent will consider that he's
            training and will select a node to explore.
        """
        assert isinstance(state, np.ndarray)
        if self.default_state is None:
            self.default_state = state.copy()
        if test_episode:
            assert isinstance(goal, np.ndarray) and goal.shape == self.goal_shape
        super().start_episode(state, test_episode=test_episode)

        # Reset episodes variables if we already made an episode before
        self.last_node_passed = None
        self.global_path = None
        self.last_exploration_trajectory = []
        self.current_subtask_steps = 0
        self.done = False
        self.under_exploration = False

        # Start episode
        self.learn = goal is None
        if len(self.reachability_graph.nodes()) == 0:
            self.create_node(state)  # Create node on state with id=0 for reachability graph initialisation
        if not self.learn:
            assert isinstance(goal, np.ndarray)
            self.final_goal = goal
        self.set_global_path(state, node_to_explore=node_to_explore)
        self.set_next_way_point()
        self.control_policy.start_episode(state, self.next_way_point, test_episode=True)
        if self.verbose:
            print("New episode. Learn = ", self.learn, ". Selected next node = " + str(self.next_way_point))

    def action(self, state, explore=True):
        return self.control_policy.action(state, explore=False)  # Assume the control policy has been trained before.

    def process_interaction(self, action, reward, new_state, done, learn=True):
        self.under_exploration = self.learn and (
            (len(self.global_path) == 1 and self.select_goal_for_exploration) or
            (len(self.global_path) == 0 and not self.select_goal_for_exploration)
        )
        # if self.under_exploration and self.last_exploration_trajectory == []:
        #     print("Starting exploration.")

        if self.under_exploration:
            self.last_exploration_trajectory.append(new_state)

        reached = False
        if self.global_path:
            reached = self.reached(new_state, self.next_way_point)
            reward = 1.0 if reached else 0.0  # Will not be used if self.re_usable_policy == True
            self.control_policy.process_interaction(action, reward, new_state, done=reached,
                                                    learn=not self.re_usable_policy)

        if self.global_path and reached:
            # The next sub-goal have been reached, we can remove it and continue to the next one
            self.current_subtask_steps = 0
            self.control_policy.stop_episode()
            passed_node = self.global_path.pop(0)
            if isinstance(passed_node, int):
                if self.last_node_passed is not None and self.learn:
                    self.on_edge_crossed(self.last_node_passed, passed_node)
                self.last_node_passed = passed_node
            if self.global_path:
                # Get next goal
                self.set_next_way_point()

                # TODO: ENV
                # if new_state[0] < 0 and new_state[1] > 0 and len(self.global_path) > 1:
                #     if (self.next_way_point[0] > 0 and self.next_way_point[1]) > 0 or \
                #             (self.next_way_point[0] < 0 and self.next_way_point[1] < 0):
                #         a = 1  # breakpoint

                self.control_policy.start_episode(new_state, self.next_way_point, test_episode=True)

                # TODO: ENV
                # self.environment.set_node("node_2", rgba="0 0 1 1", position=new_state[:3])
                # self.environment.set_node("node_3", rgba="0 1 0 1", position=self.next_way_point[:3])

            elif learn and self.select_goal_for_exploration:
                assert self.under_exploration
                self.extend_graph()
                self.done = True
                # else, agent continue while the main don't consider we reached the goal.

            if self.verbose:
                if len(self.global_path) == 1:
                    print("Path is done," + " exploring." if self.learn else " reaching final goal.")
                else:
                    print("Reached a way point. Next one is " + str(self.global_path[0]) + ".")
        else:
            self.current_subtask_steps += 1
            max_subtask_steps = self.exploration_duration if self.under_exploration else self.max_steps_to_reach
            if self.current_subtask_steps >= max_subtask_steps:
                graph_extended_after_exploration = not self.under_exploration
                if self.under_exploration:
                    # We Complete exploration.
                    assert not self.under_test and self.learn
                    self.extend_graph()
                    graph_extended_after_exploration = True
                    if self.select_goal_for_exploration:
                        self.control_policy.stop_episode()
                    self.done = True
                elif self.last_node_passed is None:
                    # we failed while reaching our first node. Stop episode.
                    self.done = True
                elif self.global_path and isinstance(self.last_node_passed, int) \
                        and isinstance(self.global_path[0], int):
                    # We failed crossing an edge.
                    self.on_edge_failed(self.last_node_passed, self.global_path[0], new_state)
                else:
                    # we failed to reach the test_goal after we succeed to complete our subgoals path.
                    self.done = True
                assert graph_extended_after_exploration
                if self.verbose:
                    print("We failed reaching this way point ... We're done with this episode.")
            elif self.verbose:
                print("Trying to reach way point " + str(self.next_way_point) + ". Time steps left = "
                      + str(max_subtask_steps - self.current_subtask_steps))

        # Increment the counter of the node related to 'new_state'.
        super().process_interaction(action, reward, new_state, done)
        if self.verbose:
            print("Interaction: observation=" + str(self.last_state) + ", action=" + str(action) + ", new_state="
                  + str(new_state) + ", agent goal=" + str(self.control_policy.current_goal))

    def set_next_way_point(self):
        if not self.global_path:
            self.next_way_point = None
        elif isinstance(self.global_path[0], np.ndarray):
            self.next_way_point = self.global_path[0].copy()
        elif isinstance(self.global_path[0], int):
            self.next_way_point = self.get_node_attribute(self.global_path[0], "state").copy()

    def on_edge_crossed(self, last_node_passed, next_node_way_point):
        pass

    def on_edge_failed(self, last_node_passed, next_node_way_point, new_state):

        # print("Failed to reach ", next_node_way_point, " from ", last_node_passed)
        if self.learn and not self.under_test:
            edge_attributes = self.reachability_graph.edges[last_node_passed, next_node_way_point]
            edge_attributes["cost"] = float("inf")
            edge_attributes["pruned_reason"] = "upside_down" if new_state.shape[-1] > 3 and new_state[2] < 0.3 else "unknown"

            # Replanify
            # self.start_episode(self.last_state, node_to_explore=self.exploration_node)
            self.done = True
        else:
            self.done = True

    """
    PLANNING FUNCTIONS 
    """
    def reached(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return np.linalg.norm(state[self.state_to_goal_filter] - goal) < self.tolerance_radius

    def shortest_path(self, node_from, node_to_reach):
        try:
            path_weight, paths = nx.bidirectional_dijkstra(self.reachability_graph, node_from, node_to_reach, "cost")
            if path_weight == float("inf"):
                return []
            else:
                return paths
        except NetworkXNoPath:
            return []

    def get_path_to(self, state, goal) -> list:
        """
        Use the information stored about the environment to compute a global path from the given state to the given
        goal.
        """
        node_from = self.get_node_for_state(state)
        node_to = self.get_node_for_state(goal)
        return self.shortest_path(node_from, node_to)

    def sample_exploration_target(self, explored_node):

        angle = random() * 2 * math.pi - math.pi
        if isinstance(explored_node, int):
            last_node_explored_state = self.get_node_attribute(explored_node, "state")
        else:
            last_node_explored_state = explored_node
        # Select a target
        x_diff = math.cos(angle) * self.exploration_goal_range
        y_diff = math.sin(angle) * self.exploration_goal_range
        target_goal = last_node_explored_state.copy()
        target_goal[:2] += [x_diff, y_diff]
        self.sampled_exploration_goal = target_goal.copy()
        return target_goal

    """
    ENVIRONMENT EXPLORATION / GRAPH BUILDING FUNCTIONS
    """
    def set_global_path(self, state, node_to_explore=None):
        """
        Set the global path. It will depend on whether we are exploring or not.
        """
        self.global_path = []
        if not self.reachability_graph.nodes:
            raise Exception("Agent's graph hasn't been initialised properly.")

        # Get the node with the lowest number of explorations from it, or reached counter.
        self.node_from = self.get_node_for_state(state)
        if self.learn:
            # Find the closest node g_0 as our first sub_goal
            g_0_state = nx.get_node_attributes(self.reachability_graph, "state")[self.node_from]
            distance_to_g0 = self.get_normalised_distance_estimation(state, g_0_state)
            if distance_to_g0 > self.edges_distance_threshold:
                self.global_path = []  # Impossible to join another graph than our current one.
            else:
                self.exploration_node = node_to_explore
                if self.exploration_node is None:
                    self.exploration_node = min(self.reachability_graph.nodes(data=True), key=lambda x: x[1]["explorations"])[0]
                    self.reachability_graph.nodes[self.exploration_node]["explorations"] += 1

                # Find the shortest path through our reachability graph to the best node
                self.global_path = self.shortest_path(self.node_from, self.exploration_node)
        else:
            node_from = self.get_node_for_state(state)
            target_node = self.get_node_for_state(self.final_goal)

            # Find the shortest path through our reachability graph to the best node
            self.global_path = self.shortest_path(node_from, target_node)

        if self.learn and self.select_goal_for_exploration:
            exploration_node_state = self.global_path[-1] if self.global_path else self.last_state[self.state_to_goal_filter]
            self.global_path.append(self.sample_exploration_target(exploration_node_state))
        elif not self.learn:
            self.global_path.append(self.final_goal)

    """
    GRAPH EXPLOITATION FUNCTIONS
    """
    def get_node_attribute(self, node_id, attribute):
        attributes = nx.get_node_attributes(self.reachability_graph, attribute)
        return attributes.get(node_id)

    def get_node_for_state(self, state, data=False):
        """
        Select the node that best represent the given state.
        """
        assert isinstance(state, np.ndarray)
        if not self.reachability_graph.nodes:
            return None  # The graph  hasn't been initialised yet.
        if state.shape[0] == self.goal_size and self.goal_size < self.state_size:
            state = np.concatenate((state, self.default_state[state.shape[0]:]))
        node_data = None
        closest_distance = None
        for node_id, args in self.reachability_graph.nodes(data=True):
            distance = self.control_policy.get_estimated_distances(state, args["state"])
            if closest_distance is None or distance < closest_distance:
                node_data = (node_id, args)
                closest_distance = distance
        return node_data if data else node_data[0]

    def get_normalised_distance_estimation(self, states, goals):
        """
        Use the UVFA to get a value function approximation between two states.
        """

        # Convert inputs to batch
        if len(states.shape) == 1:
            states = states[np.newaxis]
        if len(goals.shape) == 1:
            goals = goals[np.newaxis]

        # Operate projections: States should be in the state space and goals in the sub-goal space.
        # - Inverse projection G -> S if the states don't have the right size
        #   (computing distance between goal and state).
        if states.shape[-1] != self.state_size:
            if states.shape[-1] != self.goal_size:
                raise ValueError("Unknown state shape.")
            states = self.goal_to_state(states)

        # - Projection S -> G if the given sub-goals are actually states or goals. Our control policy only know
        #   goals, not goals.
        if goals.shape[-1] != self.goal_size:
            if goals.shape != self.state_size:
                raise ValueError("Unknown goal shape")
            goals = goals[self.state_to_goal_filter]

        # Now inputs have the right size. We can compute the distance.
        estimated_distance = self.control_policy.get_estimated_distances(states, goals)

        # Returns the normalised estimated distance (assuming self.distance_estimation_min == 0)
        assert isinstance(estimated_distance / self.distance_estimation_max, np.ndarray)
        return estimated_distance / self.distance_estimation_max

    def get_reachable_from(self, node):
        """
        return a set of nodes that are reachable from the given node
        :param node:
        :return:
        """
        return list(self.reachability_graph.neighbors(node))

    """
    GRAPH BUILDING FUNCTIONS
    Some of them can/should be override by subclasses that implement different graph building strategies.
    """
    def create_node(self, state, **params):
        assert len(state.shape) == 1, "A node cannot be created from states batch"
        attributes = copy.deepcopy(self.nodes_attributes)
        attributes["explorations"] = 0
        for key, value in params.items():
            attributes[key] = value
        for key, value in attributes.items():
            if isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
                # Here, the value of this parameter should be initialised using a function call.
                # The value inside self.nodes_attributes is a tuple, with the function in first attribute, and it's
                # parameters as a dict in the second.
                # Ex: self.create_node(weights, {model: (initialise_neural_network, {layer_1_size: 200})}
                #   will do attributes[model] = initialise_neural_network(layer_1_size=200)
                function = value[0]
                parameters_dict = value[1]
                attributes[key] = function(**parameters_dict)

        node_id = self.higher_node_id + 1
        self.higher_node_id += 1
        attributes["state"] = state[self.state_to_goal_filter]
        # NB: State is the state that belong to this node. Because it's a topological graph, every node have a position
        # in the state space that can be associated with a state. In neural gas, the word 'weights' is used in reference
        # to neurons weights. But we prefer to use the word 'state' to avoid any ambiguity with path finding cost of
        # taking a node in path finding algorithms. We use the word 'cost' on edges for that.
        self.reachability_graph.add_node(node_id, **attributes)
        return node_id

    def create_edge(self, first_node, second_node, **params):
        """
        Create an edge between the two given nodes. If potential is True, it means that the edge weight will be lower in
        exploration path finding (to encourage the agent to test it) or higher in go to mode (to bring a lower
        probability of failure).
        """
        assert (first_node, second_node) not in self.reachability_graph.edges
        attributes = copy.deepcopy(self.edges_attributes)
        for key, value in params.items():
            attributes[key] = value
        attributes["cost"] = params.get("cost", 1.)  # Set to one only if it's unset.
        self.reachability_graph.add_edge(first_node, second_node, **attributes)

    def extend_graph(self):
        """
        Update the reachability graph using the exploration trajectory.
        Precondition: An exploration trajectory has been made.
        """
        assert not self.under_test
        assert self.last_exploration_trajectory != []

        # Build nodes + explored states batch
        explored_states = np.array(self.last_exploration_trajectory)
        explored_goals = explored_states[:, self.state_to_goal_filter]
        explored_states = self.goal_to_state(explored_goals)


        nodes_goals = np.array(list(nx.get_node_attributes(self.reachability_graph, "state").values()))
        nodes_states = self.goal_to_state(nodes_goals)
        nb_explored_states = len(self.last_exploration_trajectory)
        nb_nodes = nodes_states.shape[0]

        explored_states = np.repeat(explored_states, nb_nodes, axis=0)
        explored_goals = np.tile(explored_goals, (nb_nodes, 1))
        nodes_states = np.repeat(nodes_states, nb_explored_states, axis=0)
        nodes_goals = np.tile(nodes_goals, (nb_explored_states, 1))

        forward_distance_estimations = self.get_normalised_distance_estimation(nodes_states, explored_goals)
        backward_distance_estimations = self.get_normalised_distance_estimation(explored_states, nodes_goals)
        for state_id, state in enumerate(self.last_exploration_trajectory):
            links_to_do = []

            # Select relevant nodes for this state (that are not too far so the computed distance is relevant, because
            # data are not too different from what our agent saw in his pre-training).

            # Get close node's ID
            nodes_goals_ = np.array(list(nx.get_node_attributes(self.reachability_graph, "state").values()))
            relevant_nodes_ids = np.squeeze(np.argwhere((np.absolute(nodes_goals_ - state[self.state_to_goal_filter])
                                                         <= self.s_g_max_dist).all(-1)), -1).tolist()

            # Iterate through them
            # for node_id in self.reachability_graph.nodes(data=False):
            for node_id in relevant_nodes_ids:

                forward_estimated_distance = forward_distance_estimations[node_id * nb_explored_states + state_id]
                backward_estimated_distance = backward_distance_estimations[state_id * nb_nodes + node_id]

                if forward_estimated_distance < self.nodes_distance_threshold and \
                        backward_estimated_distance < self.nodes_distance_threshold:
                    break
                if forward_estimated_distance < self.edges_distance_threshold:
                    links_to_do.append({"node": node_id, "forward": True, "cost": forward_estimated_distance})
                if backward_estimated_distance < self.edges_distance_threshold:
                    links_to_do.append({"node": node_id, "forward": False, "cost": backward_estimated_distance})
            else:
                # => this observation is far enough from any nodes
                # Create node
                new_node = self.create_node(state)

                new_node_state_batch = np.repeat(state[np.newaxis], nb_explored_states, axis=0)
                new_node_goal_batch = new_node_state_batch[:, self.state_to_goal_filter]
                explored_states_array = np.array(self.last_exploration_trajectory)

                new_node_forward_distance_estimations = self.get_normalised_distance_estimation(
                    new_node_state_batch, explored_states_array[:, self.state_to_goal_filter])
                new_node_backward_distance_estimations = \
                    self.get_normalised_distance_estimation(explored_states_array, new_node_goal_batch)

                forward_distance_estimations = np.concatenate((forward_distance_estimations,
                                                               new_node_forward_distance_estimations))
                backward_distance_estimations = np.insert(backward_distance_estimations,
                                                          np.arange(1, nb_explored_states + 1) * nb_nodes,
                                                          new_node_backward_distance_estimations)
                nb_nodes += 1

                for link_to_do in links_to_do:
                    if link_to_do["forward"]:
                        self.create_edge(link_to_do["node"], new_node, cost=link_to_do["cost"])
                    else:
                        self.create_edge(new_node, link_to_do["node"], cost=link_to_do["cost"])

    def remove_graph_around(self, node):
        """
        Remove the sub-graph around the given node. Should be called if a part of the graph is isolated.
        """
        to_remove = [n for n in self.reachability_graph.neighbors(node)]
        self.reachability_graph.remove_node(node)
        for node in to_remove:
            self.remove_graph_around(node)

    def remove_node(self, node_id, remove_isolated_nodes=True):
        """
        Remove a node in the graph. Ensure that every node are still reachable from the initial node.
        :return: list of nodes additionally removed
        """
        neighbors = self.reachability_graph.neighbors(node_id)
        self.reachability_graph.remove_node(node_id)
        if remove_isolated_nodes:
            removed_nodes = []
            for node_id in copy.deepcopy(neighbors):
                try:
                    nx.shortest_path(self.reachability_graph, 0, node_id)
                except NetworkXNoPath:
                    removed_nodes.append(node_id)
                    self.remove_graph_around(node_id)
            return removed_nodes
        return []

    def remove_edge(self, node_1, node_2):
        # Remove the edge
        try:
            self.reachability_graph.remove_edge(node_1, node_2)
        except NetworkXError:
            return

        # If this operation isolated a part of the graph from the start point, remove the isolated sub-graph
        try:
            nx.shortest_path(self.reachability_graph, 0, node_1)
        except NetworkXNoPath:
            self.remove_graph_around(node_1)
        try:
            nx.shortest_path(self.reachability_graph, 0, node_2)
        except NetworkXNoPath:
            self.remove_graph_around(node_2)

    def goal_to_state(self, goals):
        goals_ = goals[np.newaxis] if len(goals.shape) == 1 else goals
        states = np.tile(self.default_state, (goals_.shape[0], 1))
        states[:, self.state_to_goal_filter] = goals_
        return states

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "control_policy" or (k == "control_policy_value_based_agent_class" and isinstance(v, Agent)):
                setattr(result, k, self.control_policy.copy())
            elif k == "init_params":
                new_dict = {}
                for k_, v_ in v.items():
                    new_dict[k_] = copy.deepcopy(v_) if k_ != "control_policy" else v_.copy()
                setattr(result, k, new_dict)
            # elif k == "environment":  # TODO: ENV
            #     setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def reset(self):
        # self.__init__(self.control_policy_goal_conditioned_wrapper, self.control_policy_value_based_agent_class,
        #               self.state_space, self.action_space, environment=self.environment, **self.init_params)
        # TODO: remove ENV
        self.__init__(self.control_policy_goal_conditioned_wrapper, self.control_policy_value_based_agent_class,
                      self.state_space, self.action_space, **self.init_params)

    def save(self, directory):
        if directory[-1] != "/":
            directory += "/"
        create_dir(directory)

        # Save useful data to rebuild the class
        with open(directory + "control_policy_goal_conditioned_wrapper.pkl", 'wb') as f:
            pickle.dump(self.control_policy_goal_conditioned_wrapper, f)
        with open(directory + "control_policy_value_based_agent_class.pkl", 'wb') as f:
            pickle.dump(self.control_policy_value_based_agent_class, f)
        with open(directory + "state_space.pkl", 'wb') as f:
            pickle.dump(self.state_space, f)
        with open(directory + "action_space.pkl", 'wb') as f:
            pickle.dump(self.action_space, f)

        # Recursive save on our control policy
        self.control_policy.save(directory + "control_policy/")

        # Save graph
        with open(directory + "reachability_graph.pkl", "wb") as f:
            pickle.dump(self.reachability_graph, f)

        # Save data computed after pre-training
        with open(directory + "distance_estimation_max.pkl", "wb") as f:
            pickle.dump(self.distance_estimation_max, f)

        with open(directory + "higher_node_id.pkl", "wb") as f:
            pickle.dump(self.higher_node_id, f)

    def load(self, directory):
        if directory[-1] != "/":
            directory += "/"
        assert os.path.isdir(directory)

        # Save useful data to rebuild the class
        with open(directory + "control_policy_goal_conditioned_wrapper.pkl", 'rb') as f:
            self.control_policy_goal_conditioned_wrapper = pickle.load(f)
        with open(directory + "control_policy_value_based_agent_class.pkl", 'rb') as f:
            self.control_policy_value_based_agent_class = pickle.load(f)
        with open(directory + "state_space.pkl", 'rb') as f:
            self.state_space = pickle.load(f)
        with open(directory + "action_space.pkl", 'rb') as f:
            self.action_space = pickle.load(f)

        # Recursive save on our control policy
        self.control_policy.load(directory + "control_policy/")

        # Save graph
        with open(directory + "reachability_graph.pkl", "rb") as f:
            self.reachability_graph = pickle.load(f)

        # Save data computed after pre-training
        with open(directory + "distance_estimation_max.pkl", "rb") as f:
            self.distance_estimation_max = pickle.load(f)

        with open(directory + "higher_node_id.pkl", "rb") as f:
            self.higher_node_id = pickle.load(f)
