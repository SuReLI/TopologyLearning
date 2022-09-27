import copy
from enum import Enum

import networkx as nx
import numpy as np

from agents.discrete.goal_conditioned_dqn_her import DQNHERAgent
from agents.discrete.goal_conditioned_dqn_her_diff import DqnHerDiffAgent
from agents.agent import Agent
import settings


class TopologyLearnerMode(Enum):
    LEARN_ENV = 1
    GO_TO = 2
    PATROL = 3  # Not implemented. The ability of patrolling is a good metric to study. It's the major advantage of
    # A graph learned with GWR over a graph shaped as a tree.


"""
Sub-classes to implement:
 - Graph building strategies
    > GWR
    > AVEC NN
    > DISTANCE UNIQUEMENT ?
    > Q-VALUE ?
 - COUNT-BASED SELECTION
"""


class PlanningTopologyLearner(Agent):
    """
    An agent that is trained to learn the environment topology, so that learn by interacting with its environment, but
    don't need to reach a goal to do so. Then, he is able to exploit his knowledge of his environment to reach goals or
    to patrol inside it.

    self.mode is used to make the agent know what he should do during this episode.
    """

    def __init__(self, **params):
        """
        NB: The environment is given to allow the agent to generate environment image, so we can generate video to
        observe what he does between two waypoints for example. It is not and shouldn't be used to learn.
         - max_steps_per_edge: In any mode, define the number of time steps the agent has to reach the next sub-goal.
         Passed this duration, reaching the next goal is considered failed (look inside
         self.on_reaching_waypoint_failed to see the consequences).
         - random_exploration_duration: number of time steps during which we will explore using random actions, once our
         last sub-goal is reached and during an exploration episode.
         - trial_out_graph: In GO_TO mode, this variable defines the number of time steps during which the agent can try
         to reach the final goal once the  last sub-goal is reached.
         - topology_manager_class: Class of this agent topology building strategy.
         - nodes_attributes, edges_attributes: default attributes to associate with node/edged on their creation.
        """

        # ALGORITHM TYPE
        # re_usable_policy=True implies a pre-training to the given policy, and that this policy will not be trained
        # during learning..
        self.re_usable_policy = params.get("re_usable_policy", True)

        # MDP ARGUMENTS
        self.environment = params.get("environment")
        self.action_space = params.get("action_space")
        self.state_space = params.get("state_space")

        # Margin distance under which a goal can be considered reached
        # Ex.: s = (x, y), g = (x', y'), tolerance_margin = (a, b)
        # goal is reached by state s if x' - x <= a and y' - y <= b.
        self.tolerance_radius = params.get("tolerance_radius", .1)

        # GRAPH BUILDING HYPER-PARAMETERS
        self.nodes_attributes = params.get("nodes_attributes", {})
        self.edges_attributes = params.get("edges_attributes", {})
        #  -> Default nodes and edges attributes on creation as a dict, like {nb_explorations: 0} for nodes.
        self.random_exploration_duration = params.get("random_exploration_duration", 90)
        #  -> Duration of any explorations from a node we want to explore from, in number of interactions.

        # SUB-GOALS PLANNING ARGUMENTS
        self.max_steps_to_reach = params.get("max_steps_to_reach", 50)

        # MISC. ARGUMENTS
        self.device = params.get("device", settings.device)
        params["name"] = params.get("name", "Topology Learner")
        self.verbose = params.get("verbose", False)

        super().__init__(**params)

        # LOW-LEVEL ACTIONS PLANNING ARGUMENTS

        """
        General attributes (for any mode)
        """
        self.mode = TopologyLearnerMode.LEARN_ENV
        self.state_to_goal_filter = params.get("state_to_goal_filter", [1, 1] + [0 for _ in range(self.state_size - 2)])
        self.state_to_goal_filter = np.array(self.state_to_goal_filter).astype(np.bool)
        self.goal_size = np.where(self.state_to_goal_filter == True)[0].shape[0]
        default_goal_reaching_agent = DQNHERAgent(state_space=self.state_space, action_space=self.action_space,
                                                  device=self.device, state_to_goal_filter=self.state_to_goal_filter)
        self.goal_reaching_agent = params.get("goal_reaching_agent", default_goal_reaching_agent)
        self.oriented_graph = params.get("oriented_graph", True)
        self.topology = nx.DiGraph() if self.oriented_graph else nx.Graph()
        self.next_node_way_point = None

        # The number of episode will depend on the task we are doing. In any mode, the agent choose when he's done doing
        # the task.
        #   For exploration, our agent will continue its environment while he didn't finish his exploration strategy.
        #   For goal reaching mode, our agent will continue until he reached the goal or until he considers that he
        # failed to reach it.
        #   For patrol, our agent will fix a custom time limit.
        self.done = False

        # In any mode, we will try to follow trajectories inside the learned topology. Doing so, if we failed reaching
        # the next node, the entire path is compromised. To make sure we don't try infinitely to reach an unreachable
        # sub-goal, we should count how many time steps we tried to reach it so far, to detect it as an impossible
        # subtask. This allows us to break and start a new episode instead of looping infinitely.
        self.current_subtask_steps = 0  # Steps done so far to reach the next node.
        self.last_node_passed = None  # So we can know which edge is concerned if it failed.
        self.next_goal = None

        """
        EXPLORATION AND TOPOLOGY BUILDING ATTRIBUTES
            Explorations steps are simple:
             - Select a node to explore from (the one less chosen for exploration)
             - Find a path to this node in the topological graph using A*, Dijikstra, or any path finding algorithm,
             - For each waypoint, try to reach it, until we don't.
             - Once we reached the last sub-goal (aka. The node initially selected for exploration), perform randoms
             actions for a fixed duration, to sample random states next to this node.  
             - Use the randomly sampled states to add new nodes to our graph.
        """
        self.last_node_explored = None
        self.last_exploration_trajectory = []  # Trajectory made once we reached last exploration waypoint.
        # Once we reached a node that has been selected as interesting for exploration, we will explore using a random
        # policy for a fixed duration.
        self.random_exploration_steps_left = None
        self.higher_node_id = -1  # Id of the younger node. To know which id to give to a new node.
        self.current_exploration_nodes_path = None
        self.explored_node_choice_criteria = "explorations"  # Can be "explorations" or "reached"

        """
        GOAL REACHING ATTRIBUTES
            In goal reaching tasks, our agent will choose a path (a list of interesting nodes) to reach this goal. Once 
            he reached the last one, he will have a fixed number of interactions to reach the goal. Once this amount of 
            interaction is exceeded, the agent will consider that the goal is not reachable, and can work on another 
            task.
        """
        # How many interaction we did to reach the goal since the last node in path:
        # Counter of how many times steps left before to consider we failed reaching the goal
        self.nb_trial_out_graph_left = None
        self.final_goal = None
        self.current_goal_reaching_nodes_path = None

    """
    MDP LIFECYCLE FUNCTIONS
    """

    def on_simulation_start(self):
        self.goal_reaching_agent.on_simulation_start()

    def on_episode_start(self, *args):
        """
        args shape: (state, mode, goal) but goal can be removes if we're not testing goal reaching ability.

        EnvironmentLearnerMode.LEARN_ENV: Our agent select a goal to reach, and explore his environment by reaching it.
            The idea is that the goal to reach is inside an area that is promising for exploration.

        EnvironmentLearnerMode.GO_TO: Our agent exploit his topology to reach the goal. Our agent will NOT learn
            anything during this task.
        """

        state = args[0]
        super().on_episode_start(state)

        # Start episode
        state, self.mode = args[:2]
        if self.mode == TopologyLearnerMode.LEARN_ENV:
            if len(self.topology.nodes()) == 0:
                self.create_node(state)  # Create node on state with id=0 for topology initialisation
            self.set_exploration_path(state)
        elif self.mode == TopologyLearnerMode.GO_TO:
            _, _, goal = args
            fill = np.zeros(self.state_space.shape[0] - 2)
            self.final_goal = np.concatenate((goal, fill))[self.state_to_goal_filter]
            # self.final_goal = goal
        self.next_node_way_point = self.get_next_node_waypoint()
        if self.next_node_way_point is None:
            # We started on the node to reach. Happen when our graph have a single node.
            # The following function should make our agent explore if it's in the right mode.
            self.on_path_done(state)
        else:
            self.next_goal = self.get_goal_from_node(self.next_node_way_point)
            self.goal_reaching_agent.on_episode_start(state, self.next_goal)
        if self.verbose:
            print("New episode. Mode = ", self.mode.name, " Selected next node = " + str(self.next_node_way_point))

    def action(self, state):
        if self.random_exploration_steps_left is not None:
            return self.action_space.sample()
        return self.goal_reaching_agent.action(state)

    def on_action_stop(self, action, new_state, reward, done, learn=True):
        learn = self.mode == TopologyLearnerMode.LEARN_ENV
        if self.random_exploration_steps_left is not None:
            assert self.mode == TopologyLearnerMode.LEARN_ENV
            self.last_exploration_trajectory.append(new_state)
            self.random_exploration_steps_left -= 1
            if self.random_exploration_steps_left == 0:
                if self.verbose:
                    print("Finished random exploration. We're done with this episode")
                self.extend_graph()
                self.done = True
            else:
                if self.verbose:
                    print("We continue random exploration for " + str(self.random_exploration_steps_left)
                          + " more time steps.")
        elif self.nb_trial_out_graph_left is not None:
            assert self.mode == TopologyLearnerMode.GO_TO
            self.nb_trial_out_graph_left -= 1
            if self.nb_trial_out_graph_left == 0:
                if self.verbose:
                    print("We're done trying to reach the final goal.")
                self.done = True
            elif self.verbose:
                print("We continue to reach global goal for " + str(self.nb_trial_out_graph_left)
                      + " more time steps.")
            reached = self.reached(new_state, goal=self.final_goal)
            reward = 1.0 if reached else 0.0  # Will not be used if self.re_usable_policy == True
            self.goal_reaching_agent.on_action_stop(action, new_state, reward, done=reached,
                                                    learn=not self.re_usable_policy)
        else:  # We are trying to reach a sub-goal
            reached = self.reached(new_state)
            reward = 1.0 if reached else 0.0  # Will not be used if self.re_usable_policy == True
            self.goal_reaching_agent.on_action_stop(action, new_state, reward, done=reached,
                                                    learn=not self.re_usable_policy)

            while self.reached(new_state):
                reached_one = True
                # The next sub-goal have been reached, we can remove it and continue to the next one
                if self.last_node_passed is not None and learn and \
                        self.topology.edges[self.last_node_passed, self.next_node_way_point]["potential"]:
                    self.on_edge_crossed(self.last_node_passed, self.next_node_way_point)

                self.last_node_passed = self.next_node_way_point
                self.next_node_way_point = self.get_next_node_waypoint()
                self.current_subtask_steps = 0
                self.goal_reaching_agent.on_episode_stop()
                if self.next_node_way_point is None:
                    self.on_path_done(new_state)
                    if self.verbose:
                        print("Path is done.")
                else:
                    if self.verbose:
                        print("Reached a way point. Next one is " + str(self.next_node_way_point) + ".")
                    self.next_goal = self.get_goal_from_node(self.next_node_way_point)
                    self.goal_reaching_agent.on_episode_start(new_state, self.next_goal)
            if not reached:
                self.current_subtask_steps += 1
                if self.current_subtask_steps > self.max_steps_to_reach:
                    # We failed reaching the next waypoint
                    if self.last_node_passed is not None and learn:
                        self.on_edge_failed(self.last_node_passed, self.next_node_way_point)
                    if self.verbose:
                        print("We failed reaching this way point ... We're done with this episode.")
                    self.done = True
                else:
                    if self.verbose:
                        print("Trying to reach way point " + str(self.next_node_way_point) + ". Time steps left = "
                              + str(self.max_steps_to_reach - self.current_subtask_steps))

        # Increment the counter of the node related to 'new_state'.
        self.topology.nodes[self.get_node_for_state(new_state)]["reached"] += 1

        super().on_action_stop(action, new_state, reward, done)

        if self.verbose:
            print("Interaction: state=" + str(self.last_state) + ", action=" + str(action) + ", new_state="
                  + str(new_state) + ", agent goal=" + str(self.goal_reaching_agent.current_goal))

    def on_edge_crossed(self, last_node_passed, next_node_way_point):
        edge_attributes = self.topology.edges[last_node_passed, next_node_way_point]
        edge_attributes["potential"] = False
        edge_attributes["exploration_cost"] = 1.
        edge_attributes["go_to_cost"] = 1.

    def on_edge_failed(self, last_node_passed, next_node_way_point):
        edge_attributes = self.topology.edges[last_node_passed, next_node_way_point]
        edge_attributes["potential"] = False
        edge_attributes["exploration_cost"] = float('inf')
        edge_attributes["go_to_cost"] = float('inf')

    def on_episode_stop(self, learn=True):
        super().on_episode_stop()

        # Reset episodes variables
        self.last_node_passed = None
        self.nb_trial_out_graph_left = None
        self.random_exploration_steps_left = None
        self.current_goal_reaching_nodes_path = None
        self.current_exploration_nodes_path = None
        self.last_exploration_trajectory = []
        self.current_subtask_steps = 0
        self.done = False
        return self.goal_reaching_agent.on_episode_stop()

    """
    PLANNING FUNCTIONS 
    """
    def reached(self, state: np.ndarray, goal: np.ndarray = None) -> bool:
        if goal is None:
            goal = self.next_goal
        return np.linalg.norm(state[self.state_to_goal_filter] - goal) < self.tolerance_radius

    def get_next_node_waypoint(self):
        if self.mode == TopologyLearnerMode.LEARN_ENV:
            return self.get_exploration_next_node_waypoint()
        elif self.mode == TopologyLearnerMode.GO_TO:
            return self.get_go_to_next_node()
        else:
            raise Exception("Unknown mode.")

    def shortest_path(self, node_from, node_to_reach):
        if self.mode == TopologyLearnerMode.LEARN_ENV:
            attribute = "exploration_cost"
        else:
            attribute = "go_to_cost"
        return nx.shortest_path(self.topology, node_from, node_to_reach, attribute)

    def get_path_to(self, state, goal) -> list:
        """
        Use the information stored about the environment to compute a global path from the given state to the given
        goal.
        """
        node_from = self.get_node_for_state(state)
        node_to = self.get_node_for_state(goal)
        return self.shortest_path(node_from, node_to)

    def on_path_done(self, new_episode_start_state=None):
        """
        This function is called when we completed the path we chose to do inside out graph. In exploration mode,
        it means that we reached the topology border and that we can start to explore, in go_to mode, it means that
        we reached the closest node from our final goal.
        """
        if self.mode == TopologyLearnerMode.GO_TO:
            assert new_episode_start_state is not None  # We need it in this mode.
            if self.verbose:
                print("Reached last node. Trying to reach the goal for a duration of "
                      + str(self.max_steps_to_reach) + " time steps.")
            self.nb_trial_out_graph_left = self.max_steps_to_reach
            self.next_goal = self.final_goal
            self.goal_reaching_agent.on_episode_start(new_episode_start_state, self.final_goal)
        else:
            if self.verbose:
                print("Reached last node. Starting random exploration for a duration of "
                      + str(self.random_exploration_duration) + " time steps.")
            assert self.last_exploration_trajectory == []
            self.next_goal = None
            self.last_node_explored = self.last_node_passed
            assert self.last_node_explored is not None
            self.random_exploration_steps_left = self.random_exploration_duration

    def get_reachable_from(self, node):
        """
        return a set of nodes that are reachable from the given node
        :param node:
        :return:
        """
        return list(self.topology.neighbors(node))

    """
    ENVIRONMENT EXPLORATION / GRAPH BUILDING FUNCTIONS
    """
    def get_exploration_next_node_waypoint(self):
        """
        Choose and return the node we want to explore from. We select the less explored node because it's more
        efficient than the less seen node (explanation in the paper).
        """
        if self.current_exploration_nodes_path:
            return self.current_exploration_nodes_path.pop(0)
        # Otherwise, it will return None, aka. no nodes left.

    def set_exploration_path(self, state):
        """
        Select or generate a new goal that is promising for exploration.
        """
        self.current_exploration_nodes_path = []
        if not self.topology.nodes:
            return

        # Get the node with the lowest number of explorations from it, or reached counter.
        node_from = self.get_node_for_state(state)
        best_node = min(self.topology.nodes(data=True), key=lambda x: x[1][self.explored_node_choice_criteria])[0]

        # Find the shortest path through our topology to the best node
        try:
            self.current_exploration_nodes_path = self.shortest_path(node_from, best_node)
        except:
            pass
        self.topology.nodes[best_node]["explorations"] += 1

    """
    GOAL REACHING FUNCTIONS
    """
    def get_go_to_next_node(self):
        if self.current_goal_reaching_nodes_path is None:
            self.current_goal_reaching_nodes_path = []
            node_to = self.get_node_for_state(self.final_goal, reachable_only=True)
            start_node = 0 if self.last_node_passed is None else self.last_node_passed
            self.current_goal_reaching_nodes_path = self.shortest_path(start_node, node_to)
        return self.current_goal_reaching_nodes_path.pop(0) if self.current_goal_reaching_nodes_path else None

    """
    GRAPH EXPLOITATION FUNCTIONS
    """
    def get_goal_from_node(self, node_id):
        for i, args in self.topology.nodes(data=True):
            if node_id == i:
                return args["state"]

    def get_node_for_state(self, state, data=False, reachable_only=False):
        """
        Select the node that best represent the given state.
        """
        assert isinstance(state, np.ndarray)
        if state.shape[-1] == len(self.state_to_goal_filter):
            state = state[self.state_to_goal_filter]
        if not self.topology.nodes:
            return None  # The graph  hasn't been initialised yet.
        node_data = None
        closest_distance = None
        for node_id, args in self.topology.nodes(data=True):
            if reachable_only:
                try:
                    self.shortest_path(0, node_id)
                except:
                    continue
            distance = np.linalg.norm(args["state"] - state, 2)
            if closest_distance is None or distance < closest_distance:
                node_data = (node_id, args)
                closest_distance = distance
        return node_data if data else node_data[0]

    """
    GRAPH BUILDING FUNCTIONS
    Some of them can/should be override by subclasses that implement different graph building strategies.
    """
    def create_node(self, state, **params):
        assert len(state.shape) == 1, "A node cannot be created from states batch"
        attributes = copy.deepcopy(self.nodes_attributes)
        attributes["explorations"] = 0
        attributes["reached"] = 0
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
        self.topology.add_node(node_id, **attributes)
        return node_id

    def create_edge(self, first_node, second_node, potential=True, **params):
        """
        Create an edge between the two given nodes. If potential is True, it means that the edge weight will be lower in
        exploration path finding (to encourage the agent to test it) or higher in go to mode (to bring a lower
        probability of failure).
        """
        attributes = copy.deepcopy(self.edges_attributes)
        for key, value in params.items():
            attributes[key] = value
        if potential:
            attributes["potential"] = True
            attributes["exploration_cost"] = 0.
            attributes["go_to_cost"] = float("inf")
        self.topology.add_edge(first_node, second_node, **attributes)

    def extend_graph(self):
        # self.topology_manager.on_new_data(self.last_exploration_trajectory)
        raise NotImplementedError("The method extend graph depends on the graph building strategy and should be "
                                  "implemented inside child classes.")

    def remove_graph_around(self, node):
        """
        Remove the sub-graph around the given node. Should be called if a part of the graph is isolated.
        """
        to_remove = [n for n in self.topology.neighbors(node)]
        self.topology.remove_node(node)
        for node in to_remove:
            self.remove_graph_around(node)

    def remove_node(self, node_id, remove_isolated_nodes=True):
        """
        Remove a node in the graph. Ensure that every node are still reachable from the initial node.
        :return: list of nodes additionally removed
        """
        neighbors = self.topology.neighbors(node_id)
        self.topology.remove_node(node_id)
        if remove_isolated_nodes:
            removed_nodes = []
            for node_id in copy.deepcopy(neighbors):
                try:
                    nx.shortest_path(self.topology, 0, node_id)
                except:
                    removed_nodes.append(node_id)
                    self.remove_graph_around(node_id)
            return removed_nodes
        return []

    def remove_edge(self, node_1, node_2):
        # Remove the edge
        try:
            self.topology.remove_edge(node_1, node_2)
        except:
            return

        # If this operation isolated a part of the graph from the start point, remove the isolated sub-graph
        try:
            nx.shortest_path(self.topology, 0, node_1)
        except:
            self.remove_graph_around(node_1)
        try:
            nx.shortest_path(self.topology, 0, node_2)
        except:
            self.remove_graph_around(node_2)

    def copy(self):
        goal_reaching_agent = self.goal_reaching_agent.copy()
        topology = copy.deepcopy(self.topology)
        agent_copy = self.__init__(self.init_params)
        agent_copy.goal_reaching_agent = goal_reaching_agent
        agent_copy.topology = topology
        return agent_copy
