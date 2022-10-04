import copy
import os
import sys
from copy import deepcopy
from random import choice
import numpy as np
import torch
from agents.agent import Agent
import networkx as nx

from utils.sys_fun import create_dir, save_image


class SORB(Agent):
    def __init__(self, **params):
        params["name"] = params.get("name", "SORB")
        self.goal_reaching_agent = params.get("goal_reaching_agent")
        assert self.goal_reaching_agent is not None
        self.nb_nodes = params.get("nb_nodes", 200)
        self.topology = nx.Graph()
        self.final_goal = None
        self.done = False

        self.q_distance_max = None
        self.sub_goals = []

        self.tolerance_margin = params.get("tolerance_margin", (0., 0.))
        if isinstance(self.tolerance_margin, tuple):
            self.tolerance_margin = np.array(self.tolerance_margin)
        default_tolerance_radius = np.mean(self.tolerance_margin) * 1.1
        self.tolerance_radius = params.get("tolerance_radius", default_tolerance_radius)

        # Distance estimation will be normalised (like in SORB) between 0 (same state) and 1
        # (The farthest goal observed and reached during pretraining)
        self.edges_distance_threshold = params.get("edges_distance_threshold", 0.3)

        # How many interaction max before we consider that we failed to reach the next sub-goal
        self.max_interactions_per_edge = params.get("max_interactions_per_edge", 20)
        self.current_edge_interactions = 0

        # How many interaction max before we consider that we failed to reach the final goal from the last sub-goal
        self.max_final_interactions = params.get("max_final_interactions", self.max_interactions_per_edge)

        self.sub_goal_size = params.get("sub_goal_size", 2)

        self.verbose = params.get("verbose", False)
        self.last_node_reached = None

        self.current_goal = None

        super().__init__(**params)
        self.state_to_goal_filter = [True for _ in range(self.sub_goal_size)] \
                                    + [False for _ in range(self.state_size - self.sub_goal_size)]

    def init_graph(self, oracle):
        """
        Intialise a graph by sampling states in an oracle.
        :param oracle: A list of states that cover the reachable state space.
        """
        for i in range(self.nb_nodes):
            self.topology.add_node(i, state=deepcopy(choice(oracle)))

        for first_node, first_attributes in self.topology.nodes(data=True):
            for second_node, second_attributes in self.topology.nodes(data=True):
                estimates_distance = self.get_distance_estimation(first_attributes["state"], second_attributes["state"],
                                                                  normalised=True)
                if estimates_distance <= self.edges_distance_threshold:
                    self.topology.add_edge(first_node, second_node, cost=estimates_distance)

    def init_path(self, state, goal):
        start_node = self.get_node_for_state(state)
        final_node = self.get_node_for_state(goal)
        self.sub_goals = nx.shortest_path(self.topology, start_node, final_node, "cost")

    def get_node_for_state(self, state, data=False, reachable_only=True):
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
                try:  # Try to reach the node
                    nx.shortest_path(self.topology, 0, node_id)
                except:
                    continue  # Not reachable, inspect the next one.
            distance = self.get_distance_estimation(args["state"], state, normalised=False)
            if closest_distance is None or distance < closest_distance:
                node_data = (node_id, args)
                closest_distance = distance
        return node_data if data else node_data[0]

    def on_episode_start(self, *args):
        self.done = False
        self.current_edge_interactions = 0
        self.last_node_reached = None
        self.sub_goals = []
        state, self.final_goal = args
        if self.topology.nodes:
            self.init_path(state, self.final_goal)
        if self.sub_goals:
            self.current_goal = self.final_goal
        super().on_episode_start(state)
        self.goal_reaching_agent.on_episode_start(state, self.next_goal())

    def get_node_state(self, node_id):
        return self.topology.nodes()[node_id]["state"]

    def next_goal(self):
        if self.sub_goals:
            return self.get_node_state(self.sub_goals[0])
        return self.final_goal

    def on_action_stop(self, action, new_state, reward, done, learn=False):
        self.current_edge_interactions += 1
        control_agent_episode_done = False

        """
        image = self.environment.render()
        self.environment.place_point(image, self.final_goal, [255, 0, 0])
        self.environment.place_point(image, new_state, [0, 255, 0], 7)
        for sg in self.sub_goals:
            pos = self.get_node_state(sg)
            self.environment.place_point(image, self.get_node_state(sg), [0, 0, 255])
        save_image(image, self.output_directory, "img_" + str(self.episode_time_step_id))
        """

        if not learn:  # We only learn at pretraining, we will not use graph there
            # Did we have sub-goals left, and did we reach the next one?
            if self.sub_goals:
                next_sub_goal = self.get_node_state(self.sub_goals[0])
                reached = self.reached(new_state, next_sub_goal)

                if reached:
                    if self.verbose:
                        print("We reached a subgoal in ", self.current_edge_interactions, " steps", sep='')
                    self.last_node_reached = self.sub_goals.pop(0)
                    self.current_goal = self.next_goal()
                    self.current_edge_interactions = 0
                    control_agent_episode_done = True

                    # Reset a new episode for the control policy
                    self.goal_reaching_agent.on_episode_stop()
                    self.goal_reaching_agent.on_episode_start(new_state, self.current_goal)
                else:
                    if self.current_edge_interactions > self.max_interactions_per_edge:
                        if self.verbose:
                            print("We fail to reach the next sub-goal")
                        if self.last_node_reached is not None:
                            self.topology.get_edge_data(self.last_node_reached, self.sub_goals[0])["cost"] = float("inf")
                        self.done = True
            else:  # We are trying to reach the final goal after we reached every sub-goals.
                reached = (new_state == self.final_goal).all()
                if reached:
                    if self.verbose:
                        print("We reached the final goal!")
                    self.done = True
                else:
                    if self.current_edge_interactions > self.max_final_interactions:
                        if self.verbose:
                            print("We fail to reach the final goal")
                        self.done = True
        super(SORB, self).on_action_stop(action, new_state, reward, done, learn=learn)
        self.goal_reaching_agent.on_action_stop(action, new_state, reward, done or self.done or control_agent_episode_done,
                                                learn=learn)

    def reached(self, state: np.ndarray, goal: np.ndarray = None) -> bool:
        if goal is None:
            goal = self.get_node_state(self.sub_goals[0])
        dist = np.linalg.norm(goal - state, 2)
        return dist <= self.tolerance_radius

    def action(self, state):
        return self.goal_reaching_agent.action(state)

    def on_pre_training_done(self, start_state, reached_goals, oracle):
        # Compute the farthest reached goal for future q_value normalisation
        self.q_distance_max = None
        for goal in reached_goals:
            estimated_distance = self.get_distance_estimation(start_state, goal, normalised=False)
            if self.q_distance_max is None or estimated_distance > self.q_distance_max:
                self.q_distance_max = estimated_distance
        self.init_graph(oracle)

    def get_distance_estimation(self, state, goal, normalised=True):
        """
        Use the UVFA to get a value function approximation between two states.
        """
        with torch.no_grad():
            q_value_estimation = min(0, self.goal_reaching_agent.get_q_value(state, goal))
        return - q_value_estimation if not normalised else - q_value_estimation / self.q_distance_max

    def get_position(self, state):
        if state.shape[0] == 2:
            return state
        else:
            return state[:2]
    def copy(self):
        control_agent = self.goal_reaching_agent.copy()
        del self.goal_reaching_agent
        new_agent = copy.deepcopy(self)
        new_agent.goal_reaching_agent = control_agent.copy()
        self.goal_reaching_agent = control_agent.copy()
        return new_agent


