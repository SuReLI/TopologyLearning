import glob
import subprocess

environment = "grid_world"
# environment = "ant_maze"

agents_names = ["rgl", "stc", "sorb", "dqn"]
for simulation_id in range(10):
    for agent_name in agents_names:
        print("launching simulation for agent ", agent_name, " simulation n ", simulation_id, sep='')
        subprocess.Popen([environment + "/main.py", agent_name]).wait()
