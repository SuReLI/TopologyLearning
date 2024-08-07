'''
Implementations of standard RL agents:

	AgentClass: Contains the basic skeleton of an RL Agent.
	QLearningAgentClass: Q-Learning.
	LinearQAgentClass: Q-Learning with a Linear Approximator.
	RandomAgentClass: Random actor.
	RMaxAgentClass: R-Max.
	LinUCBAgentClass: Contextual Bandit Algorithm.
'''

# Grab agent classes.
from dsg_rgl_ant.simple_rl.agents.AgentClass import Agent
from dsg_rgl_ant.simple_rl.agents.FixedPolicyAgentClass import FixedPolicyAgent
from dsg_rgl_ant.simple_rl.agents.QLearningAgentClass import QLearningAgent
from dsg_rgl_ant.simple_rl.agents.DoubleQAgentClass import DoubleQAgent
from dsg_rgl_ant.simple_rl.agents.DelayedQAgentClass import DelayedQAgent
from dsg_rgl_ant.simple_rl.agents.RandomAgentClass import RandomAgent
from dsg_rgl_ant.simple_rl.agents.RMaxAgentClass import RMaxAgent
from dsg_rgl_ant.simple_rl.agents.func_approx.LinearQAgentClass import LinearQAgent
try:
	from dsg_rgl_ant.simple_rl.agents.func_approx.DQNAgentClass import DQNAgent
except ImportError:
	print("Warning: Tensorflow not installed.")
	pass

from dsg_rgl_ant.simple_rl.agents.bandits.LinUCBAgentClass import LinUCBAgent