import numpy as np
import matplotlib.pyplot as plt

from implem.settings import settings


class Node:
    def __init__(self, id, weights) -> None:
        self.id = id
        self.weights = weights
        self.neighbors = []
        self.links = []


class Link:
    def __init__(self, id, node1, node2) -> None:
        self.id = id
        self.age = 0
        self.node1 = node1
        self.node2 = node2
        if node2 not in node1.neighbors:
            node1.neighbors.append(node2)
        if node1 not in node2.neighbors:
            node2.neighbors.append(node1)
        if self not in node1.links:
            node1.links.append(self)
        if self not in node2.links:
            node2.links.append(self)
        
    def is_link(self, node1, node2):
        return (self.node1 == node1 and self.node2 == node2) or (self.node1 == node2 and self.node2 == node1)
    
    def get_nodes(self):
        return self.node1, self.node2

    nodes = property(get_nodes)


class Graph:
    def __init__(self, projection_size) -> None:
        self.next_node_id = 0  # IDs are not decremented on object deletion. Different from length.
        self.next_link_id = 0
        self.nodes = []
        self.links = []
        self.projection_size = projection_size

    def __iter__(self):
        return self.nodes.__iter__()
    
    def create_node(self, weights) -> Node:
        new_node = Node(self.next_node_id, weights)
        self.nodes.append(new_node)
        self.next_node_id += 1
        return new_node

    def get_node(self, node_id: int) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise Exception("Node not found")

    def link_nodes(self, first: Node, second: Node):
        """ Build a link between the two given nodes """
        if first in self and second in self:
            self.links.append(Link(self.next_link_id, first, second))
            self.next_link_id += 1
        else:
            raise Exception("Trying to link two nodes but at least one of them isn't inside the topology")

    def get_link_id(self, node1: Node, node2: Node):
        """ Return the link id between node1 and node2 """
        for link in node1.links:
            if node2 in link.nodes:
                return link.id
        raise Exception("Nodes are not linked!")

    def get_link(self, link_id: int) -> Link:
        for link in self.links:
            if link.id == link_id:
                return link
        raise Exception("Link not found")
    
    def remove_link(self, link_id: int, remove_alone_nodes=False):
        link = self.get_link(link_id)
        node1_id = link.node1.id
        node2_id = link.node2.id

        node1 = self.get_node(node1_id)
        node2 = self.get_node(node2_id)
        node1.neighbors.remove(node2)
        node2.neighbors.remove(node1)

        node1.links.remove(link)
        node2.links.remove(link)

        self.links.remove(link)

        if remove_alone_nodes:
            if not node1.neighbors:
                self.remove_node(node1)
            if not node2.neighbors:
                self.remove_node(node2)
    
    def remove_node(self, node: Node) -> None:
        if node.neighbors:
            for link in node.links:
                self.remove_link(link_id=link.id, remove_alone_nodes=True)
        else:
            self.nodes.remove(node)
    
    def is_linked(self, node1: Node, node2: Node):
        """ 
        Return a boolean true if nodes are linked, false otherwise. 
        Raise exception if there is a non-mutual link.
        """
        if node2 in node1.neighbors:
            if node1 in node2.neighbors:
                return True
            else:
                raise Exception("Non mutual link.")
        else:
            if node1 in node2.neighbors:
                raise Exception("Non mutual link.")
            else:
                return False
        
    def on_iteration_start(self):
        pass
        
    def on_iteration_stop(self):
        pass

    def on_new_data(self, data):
        pass

    def get_weightsmap(self):
        w = []
        for node in self:
            w.append(node.weights)
        return np.array(w)
    
    def distance(self, x: np.ndarray, node: Node):
        return np.linalg.norm(x - node.weights)

    def get_clothest_node(self, x):
        """ Return the clothest node from an input x """
        clothest = None
        distance = None
        for node in self:
            new_distance = self.distance(x, node)
            if clothest is None or distance > new_distance:
                clothest = node
                distance = new_distance
        return clothest, distance
    
    def get_clothest_pair(self, x):
        """ Return the two clothest nodes from an input x """
        first = None
        first_distance = None
        second = None
        second_distance = None
        for node in self:
            new_distance = self.distance(x, node)
            if first is None or first_distance > new_distance:
                second = first
                second_distance = first_distance
                first = node
                first_distance = new_distance
            elif second is None or second_distance > new_distance:
                second = node
                second_distance = new_distance

        return first, first_distance, second, second_distance

    # PLOT FUNCTION
    # Some function hat allow sub algorithms to edit topology plotting settings, in order to highlight some behaviours
    # For example visualise the behaviour of the topology according to edges age or nodes error.
    def get_edge_color(self, link) -> str:
        """
        Return edges color.
        link: edge for which we are looking for color,
        """
        return "#000000"

    def get_edge_width(self, link) -> float:
        """
        Return edges width.
        link: edge for which we are looking for width,
        """
        return 1.2

    def get_node_color(self, node) -> str:
        """
        Return node color.
        node: node for which we are looking for color,
        """
        return "#000000"

    def get_node_width(self, node) -> float:
        """
        Return node width.
        node: node for which we are looking for width,
        """
        return 10.

    def on_training_stop(self):
        pass

    def plot(self, figure):
        """
        Plot the topological map.
        param figure: pyplot figure where we want to plot
        param mode: node and edges color mode
        """

        # Plot weights
        for node in self.nodes:
            figure.plot(node.weights[0], node.weights[1], c=self.get_node_color(node), marker=".",
                        markersize=self.get_node_width(node))

        # Plot links
        for link in self.links:
            node1, node2 = link.nodes
            x = np.array([node1.weights, node2.weights])[:, 0]
            y = np.array([node1.weights, node2.weights])[:, 1]
            figure.plot(x, y, self.get_edge_color(link), linewidth=self.get_edge_width(link))

        figure.set_title('Topological map')
