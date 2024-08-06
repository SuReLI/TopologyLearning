import matplotlib.pyplot as plt


def plot_graph_on_environment(graph, environment, matplotlib=True, nodes_only=False):
    """
    Plot the given graph over the environment background.
    The result image is plotted in matplotlib if matplotlib == True, or is returned otherwise.
    """
    # Build image
    image = environment.render()

    # Fill image
    #  - Build nodes
    for node_id, attributes in graph.nodes(data=True):
        environment.place_point(image, attributes["state"], [125, 255, 0], width=5)

    #  - Build edges
    if not nodes_only:
        for node_1, node_2, attributes in graph.edges(data=True):
            color = [0, 255, 0]
            environment.place_edge(image, graph.nodes[node_1]["state"], graph.nodes[node_2]["state"], color, width=5)
    if matplotlib:
        plt.imshow(image)
    else:
        return image
