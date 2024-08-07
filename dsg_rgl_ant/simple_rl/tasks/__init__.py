# Python imports.
from __future__ import print_function

# Grab classes.
from dsg_rgl_ant.simple_rl.tasks.bandit.BanditMDPClass import BanditMDP
from dsg_rgl_ant.simple_rl.tasks.chain.ChainMDPClass import ChainMDP
from dsg_rgl_ant.simple_rl.tasks.chain.ChainStateClass import ChainState
from dsg_rgl_ant.simple_rl.tasks.combo_lock.ComboLockMDPClass import ComboLockMDP
from dsg_rgl_ant.simple_rl.tasks.four_room.FourRoomMDPClass import FourRoomMDP
from dsg_rgl_ant.simple_rl.tasks.gather.GatherMDPClass import GatherMDP
from dsg_rgl_ant.simple_rl.tasks.gather.GatherStateClass import GatherState
from dsg_rgl_ant.simple_rl.tasks.grid_game.GridGameMDPClass import GridGameMDP
from dsg_rgl_ant.simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from dsg_rgl_ant.simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from dsg_rgl_ant.simple_rl.tasks.hanoi.HanoiMDPClass import HanoiMDP
from dsg_rgl_ant.simple_rl.tasks.navigation.NavigationMDP import NavigationMDP
from dsg_rgl_ant.simple_rl.tasks.prisoners.PrisonersDilemmaMDPClass import PrisonersDilemmaMDP
from dsg_rgl_ant.simple_rl.tasks.puddle.PuddleMDPClass import PuddleMDP
from dsg_rgl_ant.simple_rl.tasks.random.RandomMDPClass import RandomMDP
from dsg_rgl_ant.simple_rl.tasks.random.RandomStateClass import RandomState
from dsg_rgl_ant.simple_rl.tasks.taxi.TaxiOOMDPClass import TaxiOOMDP
from dsg_rgl_ant.simple_rl.tasks.taxi.TaxiStateClass import TaxiState
from dsg_rgl_ant.simple_rl.tasks.trench.TrenchOOMDPClass import TrenchOOMDP
from dsg_rgl_ant.simple_rl.tasks.rock_paper_scissors.RockPaperScissorsMDPClass import RockPaperScissorsMDP
try:
	from dsg_rgl_ant.simple_rl.tasks.gym.GymMDPClass import GymMDP
except ImportError:
	print("Warning: OpenAI gym not installed.")
	pass
