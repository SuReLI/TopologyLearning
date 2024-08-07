'''
simple_rl
	abstraction/
		action_abs/
		state_abs/
		...
	agents/
		AgentClass.py
		QLearningAgentClass.py
		RandomAgentClass.py
		RMaxAgentClass.py
		...
	experiments/
		ExperimentClass.py
		ExperimentParameters.py
	mdp/
		MDPClass.py
		StateClass.py
	planning/
		BeliefSparseSamplingClass.py
		MCTSClass.py
		PlannerClass.py
		ValueIterationClass.py
	pomdp/
		BeliefMDPClass.py
		BeliefStateClass.py
		BeliefUpdaterClass.py
		POMDPClass.py
	tasks/
		chain/
			ChainMDPClass.py
			ChainStateClass.py
		grid_world/
			GridWorldMPDClass.py
			GridWorldStateClass.py
		...
	utils/
		chart_utils.py
		make_mdp.py
	run_experiments.py

Author and Maintainer: David Abel (cs.brown.edu/~dabel/)
Last Updated: April 23rd, 2018
Contact: dabel@cs.brown.edu
License: Apache
'''
# Fix xrange to cooperate with python 2 and 3.
try:
    xrange
except NameError:
    xrange = range

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

# Imports.
import dsg_rgl_ant.simple_rl.abstraction
import dsg_rgl_ant.simple_rl.agents
import dsg_rgl_ant.simple_rl.experiments
import dsg_rgl_ant.simple_rl.mdp
import dsg_rgl_ant.simple_rl.planning
import dsg_rgl_ant.simple_rl.tasks
import dsg_rgl_ant.simple_rl.utils
import dsg_rgl_ant.simple_rl.run_experiments

from dsg_rgl_ant.simple_rl._version import __version__