from ant_maze.environment.ant_maze import AntMaze
import local_settings
import numpy as np
import os
import matplotlib.pyplot as plt


os.environ["LD_LIBRARY_PATH"] += ":/home/disc/h.bonnavaud/.mujoco/mujoco210/bin"
os.environ["LD_LIBRARY_PATH"] += ":/usr/lib/nvidia"


def plot_trajectory(trajectory):
    environment = AntMaze(maze_name=local_settings.map_name, show=False)
    bg_image = environment.render()
    for state in trajectory:
        environment.place_point(bg_image, state, np.array([0, 0, 255]), width=20)
    plt.imshow(bg_image)
    plt.show()
