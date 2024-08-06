from enum import Enum

class AgentsIndex(Enum):
    RGL = "agents.graph_planning.tc_rgl.TC_RGL"
    REO_RGL = "agents.graph_planning.reo_rgl.REO_RGL"
    TC_RGL = "agents.graph_planning.rgl.RGL"
    DQN = "agents.discrete.goal_conditioned_dqn_her_diff.DqnHerDiffAgent"
    SAC = "agents.continuous.sac.SAC"
    SORB = "agents.global_planning.reachability_graph_learning.sorb"
    SGM = "agents.global_planning.reachability_graph_learning.sgm"
