from enum import Enum


class TileType(Enum):
    EMPTY = 0
    WALL = 1
    START = 3
    TARGET = 4


class Maze:
    def __init__(self, **params):
        self.name = params.get("name")
        assert self.name is not None, \
            "A maze cannot being instantiated without a name."
        self.map = params.get("map")
        assert self.map is not None, \
            "A maze cannot being instantiated without a map."
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.start_coordinates = params.get("start_coordinates")
        self.empty_states = []
        self.targets_coordinates = []
        for row_id, row in enumerate(self.map):
            for col_id, tile in enumerate(row):
                if tile != TileType.WALL:
                    self.empty_states.append((col_id, row_id))
                if tile == TileType.TARGET:
                    self.targets_coordinates.append((col_id, row_id))


def read_maze_file(maze_name: str) -> Maze:
    # Get the path to the maps directory
    map_directory_path = __file__[:-len("map_reader.py")]

    # Get the path to the map file to read
    map_spec_path = map_directory_path + maze_name + ".txt"

    # Read the map
    file = open(map_spec_path, "r")
    maze_array = []
    y = 0
    x, start_coordinates = None, None
    for line in file:
        if line[0] == '#':
            continue  # Ignore single line comments
        row = []
        x = 0
        for elt in line.rstrip():
            elt = int(elt)
            row.append(elt)
            if elt == TileType.START.value:
                start_coordinates = (x, y)
            x += 1
        maze_array.append(row)
        y += 1
    if not x:
        raise FileExistsError("Map file is empty.")

    # Make sure the maze spec is squared
    for line in maze_array:
        assert len(maze_array[0]) == len(line)
    return Maze(name=maze_name, map=maze_array, start_coordinates=start_coordinates)


if __name__ == "__main__":
    maze = read_maze_file("empty_square")
    print()  # This allows us to place a debug breakpoint on this line to look what's inside "maze" variable.
