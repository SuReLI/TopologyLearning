"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""
from ant_maze.control_policy.agent import LowPolicyAgentDiff
from ant_maze.environment import AntMaze
from run_HAC import run_HAC

# Instantiate the agent and Mujoco environment.
# The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.

environment = AntMaze()
agent = LowPolicyAgentDiff(environment)

# Begin training
run_HAC(environment, agent)
