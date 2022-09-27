import copy
import os.path

import numpy as np
import yaml
from gym.spaces import Box
from lxml import etree
from lxml.builder import E


def get_box(space, tile_width) -> Box:
    """
    read a space like wrote in the yaml file: aka. a dict like {low: "0, 0"; high:"5, 5"},
    and return it as a gym.spaces.Box.
    """
    if space.get("fixed") is not None:
        low = [elt * tile_width for elt in space["fixed"]]
        high = [elt * tile_width for elt in space["fixed"]]
    else:
        low = [elt * tile_width for elt in space["low"]]
        high = [elt * tile_width for elt in space["high"]]
    return Box(low=np.array(low), high=np.array(high))


def generate_xml(maze_name: str) -> (dict, str):
    """
    Generate an ant-maze environment model from a base model and maze walls description.
    :returns: (as a tuple of two elements)
     - A 'dict' that contains maze description file information.
     - A 'str' that correspond to the path where the resultant xml file has been generated.
    """

    # Load base ant-maze file etree:

    # Get the path to the current directory.
    current_directory = os.path.dirname(__file__)
    tree = etree.parse(current_directory + "/ant_maze.xml")

    # Find 'world_body' int ant-maze xml file:
    world_body_node = None
    for child in tree.getroot():
        if child.tag == "worldbody":
            world_body_node = child

    # Load walls data
    maps_directory = current_directory + "/maps/"
    try:
        with open(maps_directory + maze_name + ".yaml", 'r') as stream:
            try:
                maze_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    except:
        message = "The desired map '" + maze_name + "' does not exist. Expected map description in mujoco_files/maps/" \
                  + maze_name + ".yaml"
        raise Exception(message)
    # Build walls xml nodes using loaded walls data
    tile_size = float(maze_data["tile_size"])
    for wall in maze_data["walls"]:
        position = wall["pos"].split(" ")
        position = " ".join([str(float(elt) * tile_size) for elt in position[:2]] + position[-1:])
        size = wall["size"].split(" ")
        size = " ".join([str(float(elt) * tile_size) for elt in size[:2]] + size[-1:])
        node = E.body(
            E.geom(type="box", size=size, contype="1", conaffinity="1", rgba="0.4 0.4 0.4 1"),
            name=wall["name"], pos=position
        )
        world_body_node.append(node)
    xml_output_path = current_directory + "/generated/" + "ant_maze_" + maze_name + ".xml"
    tree.write(xml_output_path)

    # Make generated file more pretty
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(xml_output_path, parser)
    tree.write(xml_output_path, pretty_print=True)

    # Convert maze_data dict into boxes for initial_state_spaces;
    boxes = []
    for space in maze_data["initial_spaces"]:
        boxes.append(get_box(space, tile_size))
    maze_data["initial_spaces"] = copy.deepcopy(boxes)

    # ... same for reachable_spaces;
    boxes = []
    boxes_size = []  # Weights will be used by the environment to pick
    for space in maze_data["reachable_spaces"]:
        box = get_box(space, tile_size)
        boxes.append(box)
        # Append the size of the box to the weights. This is enough for weighted sampling.
        boxes_size.append(np.prod(box.high - box.low))
    maze_data["reachable_spaces"] = copy.deepcopy(boxes)
    maze_data["reachable_spaces_size"] = copy.deepcopy(boxes_size)

    return maze_data, xml_output_path


if __name__ == "__main__":
    generate_xml("empty_room")
