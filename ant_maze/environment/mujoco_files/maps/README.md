This directory contains maps description for the ant-maze environment.
They define initial state space bounds, gol space bounds, and walls position.

## Walls description
xml_generator/generate_xml(maze_name: str) (where maze_name should be the name 
of a map inside this directory) can be used to build a xml file to initialise
the simulator with walls described in the given maze file.

Walls description should be presented as follows:
``` 
walls:
 -
   name: "first wall"
   pos: "12 0 0.4"
   size: "0.25 12 0.4"
 -
   name: "second wall"
   pos: "0 12 0.4"
   size: "12.25 0.25 0.4"
```
"pos" describe the wall center position, "size" de fine the wall size in x, y, z.

 - They should be listed in a 'walls' yaml node.
 - They SHOULD (otherwise modify the generate_xml function) ave three yaml nodes: 
"name", "pos" and "size"

## Extra information
Start state space and goal spaces are only used in the AntMaze class. They are not
used to build the maze mujoco model, and so they are not includes in the xml file.

Each data should be associated to a list of Box data. Those data will be used 
to initialise many Boxes. To sample data uniformly in the concatenation of all these
spaces, we compute their weight with the area they cover.
Create many boxes instead of a single one that cover the entire maze, allow us to 
make sure that walls are not included inside these spaces. Then, we can sample 
environment coordinates, without testing if they are reachable. Position 
generation, for initial states and goals, don't need any verification loop and 
cannot be infinite.

The available_states_spaces define the set of states that can be 
reached by the agent. This variable respect the same syntax as 
initial_state_spaces and goal_spaces

```yaml
initial_state_spaces:
   - low: "-9.5, -9.5"
     high: "9.5 9.5"
state_space:
   low: "-11.5, -11.75 0 -3 -3"
  high: "11.75 11.75 1 3 3"
goal_spaces:
   - low: "-9.5 -9.5 0.45"
     high: "9.5 9.5 0.55"
```
