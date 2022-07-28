"""

Functions to make interaction and manipulation with point maze environment easier.

"""


def reset_point_maze(environment):
    environment.reset()
    goal = environment.get_target()
    initial_state = environment.reset_to_location((3, 1))
    return initial_state, goal
