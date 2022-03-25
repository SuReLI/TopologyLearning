"""
Exploration Clouds Topology (ECT) is a topology building strategy that use explorations trajectories to learn a
topology. We assume that, given an exploration trajectory, nodes that aren't directly linked to this trajectory (aka
the given trajectory goes through those nodes somewhere), shouldn't be linked to nodes, newly created from this
trajectory.

Given a trajectory we're learning with a neural gaz:
 - IGNORED NODES are those who are not directly linked to the learning trajectory.

 - INFLUENCE NODES are nodes that are directly linked to the trajectory. They can be linked to newly created
node, can be selected as a winner (so they influence newly created nodes position, so they can have new nodes next to
them), but can't move (learn) states position in the trajectory (in order to prevent the entire graph to migrate to
exploring trajectories, and move away from the agent initial position).

Ignored nodes can be linked to newly created node with a WEAK LINK. That can also be called "potential link", which is
a link that should probably exist, but should be tested before. The idea is that, using euclidian distance, we don't
know if there is a state space wall between two nodes if they are not linked by a trajectory. So we guess potential
edges, but they need to be tested first.

Latter, we will assume that some trajectories are similar, or are not separated by a state space wall. Because the
topology is 'linear' inside the group of states they represent, they will be concatenated inside an exploration cloud.
An exploration cloud is a group of state representing an exploration area. These areas are supposed to correspond to
a continuous state space. A cloud can contain states sampled from one or many trajectories.

Once a potential link (a link created between two nodes, that are close but not linked by a trajectory) is validated
as being a real link, we verify if the two exploration clouds they belong to are strongly linked. This condition is
verified for clouds C1, C2, if for each pairs of nodes n1 in C1 and n2 in C2, with a distance below a threshold,
n1 and n2 are strongly linked. The intuition is that the state space is continuous inside the area covered by these two
trajectories.
    If the latest condition is verified, the clouds C1 and C2 are concatenated and learned again (more details bellow),
otherwise, the cloud that created the confirmed link is learned again, and the node of this link that is outside the
cloud is added as an influence node to this cloud (it will not move during learning, but can be linked to learning nodes
and none cannot be created too close to this one).

In the first case, the two clouds should be concatenated. First, we remember clouds linked (strongly or weakly)
to one of the two concatenated as neighbor clouds. Then, the learning nodes from each cloud are removed.
nodes that was influence nodes for one of the two concatenated clouds, become influence nodes for the newly created
cloud. Then, states (use as GWR data) are learned by the topology using GWR, where winners can only be a node from
influence nodes and learning nodes. If there's no winners available, a new learning node is created on the selected
data. For a new data, while we are looking for the two closest nodes (inside the union of influence and learning nodes),
every node that are not candidates, but closer than one of the two winners are keep. Once the two winners are found, we
create a weak edge between winners (newly created included) and nodes that are still closer than them.

Following these rules, once a cloud is created, it will have less and less weak link while it's becoming older. It will
be concatenated to similar clouds, until neighbors clouds are separated through walls, and the agent is no more
exploring its area.
"""
import copy
import math
import random
from random import choice

import numpy as np

from agents.graph_building_strategies.topology_manager import TopologyManager


class ExplorationCloud:
    def __init__(self, cloud_id):
        self.cloud_id = cloud_id
        self.states_buffer = []
        self.influence_nodes = []
        self.learning_nodes = []


class ECT(TopologyManager):
    def __init__(self, topology, distance_function=None, nodes_attributes=None, edges_attributes=None, Sw=0.1, Sn=0.05,
                 age_max=30, activity_threshold=0.87, firing_threshold=0.05, clouds_max_size=100,
                 cloud_learn_duration=250):
        # activity_threshold=0.87 for map 7
        # activity_threshold=0.94 for map 9
        # activity_threshold=0.96 for map 10
        if edges_attributes is None:
            edges_attributes = {}
        edges_attributes["strong"] = False
        #  edges_attributes["risk"] = 1  ->  Edges don't have risk, because edges should not be created through walls.
        if nodes_attributes is None:
            nodes_attributes = {}
        nodes_attributes["win_count"] = 0
        nodes_attributes["learning_from"] = None  # Id of the cloud this node is a learning node.
        nodes_attributes["influence_of"] = []  # Ids of clouds this node is an influence node of.
        super().__init__(topology, distance_function, nodes_attributes, edges_attributes)
        self.Sw = Sw
        self.Sn = Sn
        self.age_max = age_max
        self.activity_threshold = activity_threshold
        self.firing_threshold = firing_threshold

        self.exploration_clouds = []
        self.clouds_max_size = clouds_max_size
        self.cloud_learn_duration = cloud_learn_duration
        self.next_cloud_id = 0

    def on_new_data(self, data):
        exploration_node = data[-1][0]  # Get last node
        exploration_states = data[-1][1]  # Get last exploration data

        # Verify if this exploration isn't related to any cloud.
        # TODO: revoir la condition de trajectoire inclu dans un autre, traiter le cas des explorations qui vont vers
        #  d'autres noeuds
        # + Vérifier si la trajectoire ne crée pas de lien fort

        candidates_clouds = copy.deepcopy(self.exploration_clouds)

        for cloud_index, cloud in enumerate(candidates_clouds):
            # Verify if this cloud is related
            pass  # TODO




        for state in exploration_states:
            closest_node = self.get_node_for_state(state)
            clouds_to_remove = []
            for cloud_index, cloud in enumerate(candidates_clouds):
                assert isinstance(cloud, ExplorationCloud)
                included = False
                for node in cloud.influence_nodes + cloud.learning_nodes:
                    if node == closest_node:
                        included = True
                if not included:
                    clouds_to_remove.append(cloud_index)
            for cloud_index in clouds_to_remove:
                candidates_clouds.pop(cloud_index)

        assert len(candidates_clouds) < 2
        if candidates_clouds == 1:
            cloud_id = candidates_clouds[0].cloud_id
            cloud: ExplorationCloud = self.get_cloud(cloud_id)

            # Add last node as an influence node
            cloud.influence_nodes.append(exploration_node)
            self.topology.nodes[exploration_node]["influence_of"].append(cloud.cloud_id)
            if len(exploration_states) + len(cloud.states_buffer) <= self.clouds_max_size:
                cloud.states_buffer += exploration_states
            else:
                cloud.states_buffer = random.sample(cloud.states_buffer + exploration_states, self.clouds_max_size)
        else:
            # There is no cloud fully related to this exploration samples. We should create a new one.
            new_cloud = ExplorationCloud(self.next_cloud_id)
            self.next_cloud_id += 1

            # Add last node as an influence node
            new_cloud.influence_nodes.append(exploration_node)
            self.topology.nodes[exploration_node]["influence_of"].append(new_cloud.cloud_id)

            new_cloud.states_buffer = exploration_states
            self.exploration_clouds.append(new_cloud)
            self.learn_cloud(new_cloud)

    def learn_cloud(self, cloud: ExplorationCloud):
        # Remember neighbors clouds
        neighbors_clouds = []
        for node in cloud.learning_nodes:
            for neighbor in self.topology.neighbors(node):
                clouds_to_add = self.topology.nodes[neighbor]["learning_from"] +\
                                self.topology.nodes[neighbor]["influence_of"]
                for cloud_to_add in clouds_to_add:
                    if cloud_to_add == cloud.cloud_id:
                        continue
                    if cloud_to_add not in neighbors_clouds:
                        neighbors_clouds.append(cloud_to_add)

        # Remove learning nodes
        for node in cloud.learning_nodes:
            self.topology.remove_node(node)

        # Learn the cloud topology
        for learning_step_id in range(self.cloud_learn_duration):
            state = choice(cloud.states_buffer)

            # GWR algorithm
            influence_nodes = cloud.learning_nodes + cloud.influence_nodes

            first, first_distance, first_params = (None, None, None)
            second, second_distance, second_params = (None, None, None)
            # We can choose the two best nodes.
            for node in influence_nodes:
                node_params = self.topology[node]
                node_distance = np.linalg.norm(node_params["weights"] - state, 2)
                if first is None or first_distance > node_distance:
                    second = first
                    second_distance = first_distance
                    first = node
                    first_distance = node_distance
                elif second is None or second_distance > node_distance:
                    second = node
                    second_distance = node_distance

            assert first is not None and first_params is not None
            if first in cloud.learning_nodes:
                first_params["win_count"] += 1

            # Create node if needed
            created_node = None
            activity = math.exp(- first_distance)
            first_count_ratio = math.exp(- first_params["win_count"])
            if activity < self.activity_threshold and first_count_ratio < self.firing_threshold:
                # Create a node
                created_node = self.create_node((first_params["weights"] + state) / 2)
                self.create_edge(first, created_node, strong=True)

                if second is not None:
                    self.create_edge(second, created_node, strong=True)
                    try:
                        self.topology.remove_edge(first, second)
                    except:
                        pass
            else:
                # If a new node is not added, adapt the position of the winner and its neighbors
                first_params["weights"] += self.Sw * (first_count_ratio * 4) * (state - first_params["weights"])

                if second is not None:  # Train second node
                    second_count_ratio = math.exp(- second_params["win_count"])
                    second_params["weights"] += self.Sn * (second_count_ratio * 4) * (state - second_params["weights"])
                    # Link the two best nodes
                    self.create_edge(first, second, strong=True)

            # Look for weakly linkable nodes
            if created_node is not None:
                distance_threshold = second_distance if second is not None else first_distance
                for node, node_params in self.topology.nodes(data=True):
                    if node in influence_nodes:
                        continue
                    distance = np.linalg.norm(node_params["weights"] - state, 2)
                    if distance < distance_threshold:
                        self.create_edge(node, created_node, strong=False)

    def concatenate_clouds(self, cloud1: ExplorationCloud, cloud2: ExplorationCloud):
        # Concatenate structures
        new_cloud = ExplorationCloud(self.next_cloud_id)
        self.next_cloud_id += 1

        # Concatenates data
        states_buffer = cloud1.states_buffer + cloud2.states_buffer
        if len(states_buffer) > self.clouds_max_size:
            states_buffer = random.sample(states_buffer, self.clouds_max_size)

        # Concatenate learning nodes
        new_cloud.states_buffer = states_buffer
        new_cloud.learning_nodes = cloud1.learning_nodes + cloud2.learning_nodes
        for node in new_cloud.learning_nodes:
            self.topology.nodes[node]["learning_from"] = new_cloud.cloud_id

        # Concatenate influence nodes
        for node in cloud1.influence_nodes:
            self.topology.nodes[node]["influence_of"].remove(cloud1.cloud_id)
            self.topology.nodes[node]["influence_of"].append(new_cloud.cloud_id)
        for node in cloud2.influence_nodes:
            self.topology.nodes[node]["influence_of"].remove(cloud2.cloud_id)
            self.topology.nodes[node]["influence_of"].append(new_cloud.cloud_id)

        new_cloud.influence_nodes = cloud1.influence_nodes + cloud2.influence_nodes

        self.exploration_clouds.append(new_cloud)
        self.learn_cloud(new_cloud)

    def get_cloud(self, cloud_id):
        for cloud in self.exploration_clouds:
            if cloud_id == cloud.cloud_id:
                return cloud_id

    def create_node(self, weights, learning_from=None):
        params = copy.deepcopy(self.nodes_attributes)
        if learning_from is not None:
            params["learning_from"] = learning_from
        for key, value in self.nodes_attributes.items():
            if isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
                # Here, the value of this parameter should be initialised using a function call.
                # The value inside self.nodes_attributes is a tuple, with the function in first attribute, and it's
                # parameters as a dict in the second.
                function = value[0]
                parameters_dict = value[1]
                params[key] = function(**parameters_dict)

        params["weights"] = weights
        self.topology.add_node(self.next_node_id, **params)
        self.next_node_id += 1
        return self.next_node_id - 1

    def create_edge(self, first_node, second_node, strong=None):
        params = copy.deepcopy(self.edges_attributes)
        if strong is not None:
            params["strong"] = strong
        self.topology.add_edge(first_node, second_node, **params)

