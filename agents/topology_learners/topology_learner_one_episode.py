import random
from copy import copy
from enum import Enum
from random import choice
from statistics import mean

import networkx as nx
import numpy as np

from agents import GCSACAgent
from agents.gc_agent import GoalConditionedAgent
from agents.agent import Agent
from agents.goal_conditioned_rl_agents.Continuous.continuous_actions_agent import GCAgentContinuous
from agents.goal_conditioned_rl_agents.Discrete.dqn_her import DQNHERAgent
from agents.graph_building_strategies.gwr import GWR
from agents.graph_building_strategies.topology_manager import TopologyManager
import os
import torch
import torch.nn as nn

from utils.sys_fun import generate_video

"""
self.current_agent().on_action_stop
self.current_agent().on_episode_stop
"""


def create_dir(dir_name):
    if os.path.isdir(dir_name):
        return
    dir_parts = dir_name.split("/")
    directory_to_create = ""
    for part in dir_parts:
        directory_to_create += part + "/"
        if not os.path.isdir(directory_to_create):
            try:
                os.mkdir(directory_to_create)
            except:
                print("failed to create dir " + str(directory_to_create))
                raise Exception


class TopologyLearnerMode(Enum):
    LEARN_ENV = 1
    GO_TO = 2
    PATROL = 3


class TopologyLearner(Agent):
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

        environment = params.get("environment")
        state_space = params.get("state_space")
        tolerance_radius = params.get("tolerance_radius")
        action_space = params.get("action_space")
        device = params.get("device")
        name = params.get("name", "Topology Learner")
        max_steps_per_edge = params.get("max_steps_per_edge", 50)
        random_exploration_duration = params.get("random_exploration_duration", 70)
        nb_trial_out_graph = params.get("nb_trial_out_graph", 50)
        patrol_duration = params.get("patrol_duration", 200)
        topology_manager_class = params.get("topology_manager_class", GWR)
        nodes_attributes = params.get("nodes_attributes", None)
        edges_attributes = params.get("edges_attributes", None)
        verbose = params.get("verbose", False)

        self.params = params
        super().__init__(state_space, action_space, device, name=name)

        """
        General attributes (for any mode)
        """
        self.mode = TopologyLearnerMode.LEARN_ENV
        self.topology = nx.Graph()
        self.next_node_way_point = None
        self.tolerance_radius = tolerance_radius  # Used to know if we can consider a node as reached.

        # The number of episode will depend on the task we are doing. In any mode, the agent choose when he's done doing
        # the task.
        #   For exploration, our agent will continue its environment while he didn't finish his exploration strategy.
        #   For goal reaching mode, our agent will continue until he reached the goal or until he considers that he
        # failed to reach it.
        #   For patrol, our agent will fix a custom time limit.
        self.done = False

        # In any mode, we will try to follow trajectories inside the learned topology. Doing so, if we failed reaching
        # the next node, the entire path is compromised. To make sure we don't try infinitely to reach an unreachable
        # way-point, we should count how many time steps we tried to reach it so far, to detect it as an impossible
        # subtask. This allows us to break and start a new episode instead of looping infinitely.
        self.max_steps_per_edge = max_steps_per_edge  # Maximum time steps allowed to reach the next node
        self.current_subtask_steps = 0  # Steps done so far to reach the next node.
        self.last_node_passed = None  # So we can know which edge is concerned if it failed.
        self.next_sub_goal = None

        """
        Exploration and topology building attributes
        """
        assert issubclass(topology_manager_class, TopologyManager)
        self.topology_manager = topology_manager_class(self.topology, distance_function=self.get_distance_function(),
                                                       nodes_attributes=nodes_attributes,
                                                       edges_attributes=edges_attributes)
        self.last_trajectory = [(None, [])]

        # Once we reached a node that has been selected as interesting for exploration, we will explore using a random
        # policy for a fixed duration.
        self.random_exploration_steps_left = None
        self.random_exploration_duration = random_exploration_duration
        self.last_node_explored = None

        """
        Goal reaching attributes
        """
        self.nb_trial_out_graph = nb_trial_out_graph
        # Counter of how many times steps left before to consider we failed reaching the goal
        self.nb_trial_out_graph_left = None
        self.final_goal = None
        self.go_to_current_path = None

        """
        Patrol attributes
        """
        self.patrol_duration = patrol_duration
        self.patrol_time_steps_left = self.patrol_duration

        """
        Attributes to generate outputs
        """
        self.environment = environment
        self.crossing_images = []
        self.video_id = 0
        self.output_dir = None
        # Memories to register successes for plots
        self.reaching_success_mem = []
        self.reaching_success_moving_average = []
        self.last_edge_failed = None
        self.verbose = verbose
        self.start_state = None

    def get_distance_function(self):
        """
        Return a function that return, given two states s1 and s2, - Q(s1, s2).
        """
        """
        agent = self.current_agent()
        if isinstance(agent, DQNHERAgent):
            model = agent.model
            return lambda x, y: - torch.argmax(model(np.concatenate((x, y), -1))).cpu().detach().numpy()
        if isinstance(agent, GCSACAgent):
            model = agent.critic
            return lambda x, y: - model(torch.concat((x, y), -1)).cpu().detach().numpy()
        """
        return lambda x, y: np.linalg.norm(x - y, 2)

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def append_image(self, environment, start_position, goal_position, agent_goal):
        image = environment.render()
        x, y = environment.get_coordinates(start_position)
        image = environment.set_tile_color(image, x, y, [0, 0, 255])
        x, y = environment.get_coordinates(goal_position)
        image = environment.set_tile_color(image, x, y, [0, 255, 0])
        x, y = environment.get_coordinates(agent_goal)
        image = environment.set_tile_color(image, x, y, [51, 204, 255])
        self.crossing_images.append(image)

    def create_node(self, weights):
        return self.topology_manager.create_node(weights)

    def create_edge(self, first_node, second_node):
        return self.topology_manager.create_edge(first_node, second_node)

    def on_simulation_start(self):
        output_dir = self.output_dir
        self.reset()
        self.output_dir = output_dir

    def on_episode_start(self, *args):
        """
        args shape: (state, mode, episode_id, goal) but goal can be removes if we're not testing goal reaching ability.

        EnvironmentLearnerMode.LEARN_ENV: Our agent select a goal to reach, and explore his environment by reaching it.
            The idea is that the goal to reach is inside an area that is promising for exploration.

        EnvironmentLearnerMode.GO_TO: Our agent exploit his topology to reach the goal. This task is called when
            LEARN_ENV is called, but can also be directly called with an extrinsically given goal to test ou agent or
            exploit his knowledge.

        EnvironmentLearnerMode.PATROL: initialise a long episode where we want our agent to exploit his topology to
            patrol inside his environment.
        """
        if self.verbose:
            print()
            print()
        self.last_trajectory = [(None, [])]
        self.crossing_images = []
        self.video_id += 1
        self.done = False
        self.random_exploration_steps_left = None
        self.nb_trial_out_graph_left = None
        self.go_to_current_path = None
        self.current_subtask_steps = 0
        state, self.mode, episode_id = args[:3]
        self.start_state = state
        super().on_episode_start(state, episode_id)
        if self.mode == TopologyLearnerMode.LEARN_ENV:
            if len(self.topology.nodes()) == 0:
                self.create_node(state)  # Create node on state with id=0 for topology initialisation
        elif self.mode == TopologyLearnerMode.GO_TO:
            _, _, _, goal = args
            self.final_goal = goal
        self.next_node_way_point = self.get_next_node_waypoint()
        if self.next_node_way_point is None:
            self.on_path_done(state)  # It can happen when our topology is not build yet, so we need to explore.
        else:
            self.next_sub_goal = self.get_goal_from_node(self.next_node_way_point)
            self.current_agent().on_episode_start(state, self.next_sub_goal, 0)
        if self.verbose:
            print("New episode. Mode = ", self.mode.name, " Selected next node = " + str(self.next_node_way_point))

    def reached(self, state: np.ndarray, goal: np.ndarray = None) -> bool:
        if goal is None:
            goal = self.next_sub_goal
        distance = np.linalg.norm(goal - state, 2)
        return distance < self.tolerance_radius

    def get_next_node_waypoint(self):
        if self.mode == TopologyLearnerMode.LEARN_ENV:
            next_node = self.get_exploration_next_node()
            if next_node is not None:
                self.topology.nodes[next_node]["density"] += 1
            return next_node
        elif self.mode == TopologyLearnerMode.PATROL:
            return self.get_patrol_next_node()
        elif self.mode == TopologyLearnerMode.GO_TO:
            next_node = self.get_go_to_next_node()
            return next_node
        else:
            raise Exception("Unknown mode.")

    def get_exploration_next_node(self):
        if random.random() > 0.1:
            return random.choice(list(self.topology.neighbors(self.last_node_passed)))
        # Otherwise, return None

    def get_patrol_next_node(self):
        return random.choice(list(self.topology.neighbors(self.last_node_passed)))

    def get_go_to_next_node(self):
        if self.go_to_current_path is None:
            self.go_to_current_path = []
            node_to = self.get_node_for_state(self.final_goal)
            start_node = 0 if self.last_node_passed is None else self.last_node_passed
            self.go_to_current_path = self.shortest_path(start_node, node_to)
        return self.go_to_current_path.pop(0) if self.go_to_current_path else None

    def get_path_to(self, state, goal) -> list:
        """
        Use the information stored about the environment to compute a global path from the given state to the given
        goal.
        """
        node_from = self.get_node_for_state(state)
        node_to = self.get_node_for_state(goal)
        return self.shortest_path(node_from, node_to)

    def shortest_path(self, node_from, node_to_reach):
        return self.topology_manager.shortest_path(node_from, node_to_reach)

    def current_agent(self) -> GoalConditionedAgent:
        pass

    def action(self, state):
        if self.random_exploration_steps_left is not None:
            return self.action_space.sample()
        return self.current_agent().action(state)

    def on_action_stop(self, action, new_state, reward, done, train_policy=True, learn_topology=True) -> float:
        self.last_trajectory[-1][1].append(new_state)
        if self.last_node_passed is not None:
            start_position = self.topology.nodes[self.last_node_passed]["weights"]
        else:
            start_position = self.start_state

        if self.verbose:
            print("Interaction: state=" + str(self.last_state) + ", action=" + str(action) + ", new_state="
                  + str(new_state) + ", agent goal=" + str(self.current_agent().current_goal))
        super().on_action_stop(action, new_state, reward, done)

        # Increment node density
        """
        new_node = self.get_node_for_state(new_state)
        self.topology.nodes[new_node]["density"] += 1
        """

        if self.random_exploration_steps_left is not None:
            self.random_exploration_steps_left -= 1
            if self.random_exploration_steps_left == 0:
                if self.verbose:
                    print("Finished random exploration. We're done with this episode")
                self.done = True
            else:
                if self.verbose:
                    print("We continue random exploration for " + str(self.random_exploration_steps_left)
                          + " more time steps.")
        elif self.nb_trial_out_graph_left is not None:
            self.nb_trial_out_graph_left -= 1
            if self.nb_trial_out_graph_left == 0:
                if self.verbose:
                    print("We're done trying to reach the final goal.")
                self.done = True
            elif self.verbose:
                print("We continue to reach global goal for " + str(self.nb_trial_out_graph_left)
                      + " more time steps.")
        else:
            self.append_image(self.environment, start_position, self.next_sub_goal, self.current_agent().current_goal)
            reached = self.reached(new_state)
            reward = 1 if reached else -1
            if train_policy:
                res = self.current_agent().on_action_stop(action, new_state, reward, done=reached)

            if reached:
                # The first sub-goal have been reached, we can remove it and continue to the next one
                if learn_topology:
                    self.on_reaching_waypoint_succeed(self.last_node_passed, self.next_node_way_point)
                self.last_node_passed = self.next_node_way_point
                self.next_node_way_point = self.get_next_node_waypoint()

                self.last_trajectory.append((copy(self.last_node_passed), []))
                self.current_subtask_steps = 0
                self.current_agent().on_episode_stop()
                if self.next_node_way_point is None:
                    self.on_path_done(new_state)
                else:
                    if self.verbose:
                        print("Reached a way point. Next one is " + str(self.next_node_way_point) + ".")
                    self.crossing_images = []
                    self.video_id += 1
                    self.next_sub_goal = self.get_goal_from_node(self.next_node_way_point)
                    self.current_agent().on_episode_start(new_state, self.next_sub_goal, 0)
            else:
                self.current_subtask_steps += 1
                if self.current_subtask_steps > self.max_steps_per_edge:
                    # We failed reaching the next waypoint
                    if self.verbose:
                        print("We failed reaching this way point ... We're done with this episode.")
                    self.done = True
                    if learn_topology:
                        self.on_reaching_waypoint_failed(self.last_node_passed, self.next_node_way_point)
                else:
                    if self.verbose:
                        print("Trying to reach way point " + str(self.next_node_way_point) + ". Time steps left = "
                              + str(self.max_steps_per_edge - self.current_subtask_steps))
                        print()
        if 'res' in locals():
            return res

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
                      + str(self.nb_trial_out_graph) + " time steps.")
            self.nb_trial_out_graph_left = self.nb_trial_out_graph
            self.next_sub_goal = self.final_goal
            self.current_agent().on_episode_start(new_episode_start_state, self.final_goal, 0)
        else:
            if self.verbose:
                print("Reached last node. Starting random exploration for a duration of "
                      + str(self.random_exploration_duration) + " time steps.")
            self.random_exploration_steps_left = self.random_exploration_duration
            self.last_node_explored = self.last_node_passed

    def on_reaching_waypoint_failed(self, last_node, next_node):
        print("failed between", last_node, "and", next_node)
        self.last_edge_failed = (last_node, next_node)
        self.topology_manager.on_reaching_waypoint_failed(last_node, next_node)
        self.reaching_success_mem.append(0)
        if len(self.reaching_success_mem) >= 20:
            m = mean(self.reaching_success_mem[-20:])
        else:
            m = mean(self.reaching_success_mem)
        self.reaching_success_moving_average.append(m)

        directory = self.output_dir + "cross_videos/" + str(last_node) + "_" + str(next_node) + "/"
        generate_video(self.crossing_images, directory, str(self.video_id))

    def on_reaching_waypoint_succeed(self, last_node, next_node):
        self.last_edge_failed = None
        self.topology_manager.on_reaching_waypoint_succeed(last_node, next_node)
        self.reaching_success_mem.append(1)
        if len(self.reaching_success_mem) >= 20:
            m = mean(self.reaching_success_mem[-20:])
        else:
            m = mean(self.reaching_success_mem)
        self.reaching_success_moving_average.append(m)

    def get_reachable_from(self, node):
        """
        return a set of nodes that are reachable from the given node
        :param node:
        :return:
        """
        neighbors = set([])
        for first, second in self.topology.edges():
            if first == node:
                neighbors.add(second)
            if second == node:
                neighbors.add(first)
        return list(neighbors)

    def get_goal_from_node(self, node_id):
        for i, args in self.topology.nodes(data=True):
            if node_id == i:
                return args["weights"]

    def get_node_for_state(self, state, data=False):
        """
        Select the node that best represent the given state.
        """
        assert isinstance(state, np.ndarray)
        if not self.topology.nodes:
            return None  # The graph  hasn't been initialised yet.
        closest = None
        node_data = None
        closest_distance = None
        for node_id, args in self.topology.nodes(data=True):
            distance = np.linalg.norm(args["weights"] - state, 2)
            if closest is None or distance < closest_distance:
                closest = node_id
                node_data = (node_id, args)
                closest_distance = distance
        return node_data if data else node_data[0]

    # Transfer call to the embedded goal reaching agent
    def on_episode_stop(self):
        super().on_episode_stop()
        self.topology_manager.on_new_data(self.last_trajectory)

        # Reset episodes variables
        self.last_node_passed = None
        self.nb_trial_out_graph_left = None
        self.random_exploration_steps_left = None
        self.go_to_current_path = None
        return self.current_agent().on_episode_stop()

    def reset(self):
        self.__init__(**self.params)
