# Continuous point maze

This environment implementation is a copy from the one at [SORB implementation](https://colab.research.google.com/github/google-research/google-research/blob/master/sorb/SoRB.ipynb#scrollTo=esd_jQgISaff),
that has been refactored to fit with ouf framework. Then, we can keep a main file and change the environment class
without changing anything around.

## Description

A Maze where the agent, a point in the maze, can evolve freely, and explore its environment.
The action space and the observation space are both continuous.  

## MDP details

 * action space: gym.spaces.Box(-1., 1., (2,))

| action | description         |
|:------:|---------------------|
|   0    | Evolution on axis 0 |
|   1    | Evolution on axis 1 |

 * observation space: gym.spaces.Box:
   * low = np.array([0.0, 0.0]),
   * high = np.array([maze_height, maze_width]),
    
| Observation | description        |
|:-----------:|--------------------|
|      0      | Position on axis 0 |
|      1      | Position on axis 1 |

 * Noise is 1.0 by default and can be set in the environment constructor.
It is computed by adding a gaussian noise to each action taken in the environment.
The noise value represent the std of this distribution. His mean is always 0.