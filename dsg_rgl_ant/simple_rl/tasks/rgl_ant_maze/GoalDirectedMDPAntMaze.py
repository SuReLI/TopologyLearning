from .ant_maze import AntMaze
from ...mdp.GoalDirectedMDPClass import GoalDirectedMDP


class GDMDPAntMaze(GoalDirectedMDP):
    def __init__(self, map_name, goal_state=None, use_hard_coded_events=False, seed=0, render=False):

        GoalDirectedMDP.__init__(self, range(self.env.action_space.shape[0]),
                                 self._transition_func,
                                 self._reward_func, self.init_state,
                                 salient_positions, task_agnostic=goal_state is None,
                                 goal_state=goal_state, goal_tolerance=0.6)
        
    def __init__(self, actions, transition_func, reward_func, init_state, salient_positions, task_agnostic):
        super().__init__(actions, transition_func, reward_func, init_state, salient_positions, task_agnostic)
        