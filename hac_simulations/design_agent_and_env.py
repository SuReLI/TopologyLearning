"""
This file provides the template for designing the agent and environment.  The below hyperparameters must be assigned to a value for the algorithm to work properly.
"""

import numpy as np
from environment import Environment
from utils import check_validity
from agent import Agent

def design_agent_and_env(FLAGS, model_name, reset_fixed, uniform_goal_sampling, reset_coordinates, maze_array,
                         tile_size, simulation_id, outputs_directory):

    """
    1. DESIGN AGENT

    The key hyperparameters for agent construction are

        a. Number of levels in agent hierarchy
        b. Max sequence length in which each policy will specialize
        c. Max number of atomic actions allowed in an episode
        d. Environment timesteps per atomic action

    See Section 3 of this file for other agent hyperparameters that can be configured.
    """

    FLAGS.layers = 3    # Enter number of levels in agent hierarchy

    FLAGS.time_scale = 10    # Enter max sequence length in which each policy will specialize

    # Enter max number of atomic actions.  This will typically be FLAGS.time_scale**(FLAGS.layers).  However, in the UR5 Reacher task, we use a shorter episode length.
    max_actions = 700
    # max_actions = 15

    timesteps_per_action = 15    # Provide the number of time steps per atomic action.

    # Provide initial state space consisting of the ranges for all joint angles and velocities.
    # In the Ant Reacher task, we use a random initial torso position and use fixed values for the remainder.

    initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
    initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
    initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
    initial_joint_ranges[0] = np.array([-6,6])
    initial_joint_ranges[1] = np.array([-6,6])

    # Cocatenate velocity ranges
    initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

    # Provide end goal space.
    # The code supports two types of end goal spaces if user would like to train on a larger end goal space.
    # If user needs to make additional customizations to the end goals, the "get_next_goal" method in "environment.py" can be updated.

    # In the UR5 reacher environment, the end goal will be the desired joint positions for the 3 main joints.
    max_range = 6
    goal_space_train = [[-max_range, max_range], [-max_range, max_range]]
    goal_space_test = [[-max_range, max_range], [-max_range, max_range]]


    # Provide a function that maps from the state space to the end goal space.  This is used to (i) determine whether the agent should be given the sparse reward and (ii) for Hindsight Experience Replay to determine which end goal was achieved after a sequence of actions.
    project_state_to_end_goal = lambda sim, state: state[:2]

    # Set end goal achievement thresholds.  If the agent is within the threshold for each dimension, the end goal has been achieved and the reward of 0 is granted.

    # For the Ant Reacher task, the end goal will be the desired (x,y) position of the torso
    len_threshold = 0.4
    end_goal_thresholds = np.array([len_threshold, len_threshold])


    # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal space can be the same as the state space or some other projection out of the state space.

    # The subgoal space in the Ant Reacher task is the desired (x,y,z) position and (x,y,z) translational velocity of the torso
    cage_max_dim = 8
    max_height = 1
    max_velo = 3
    height_threshold = 0.2
    subgoal_bounds = np.array([[-cage_max_dim,cage_max_dim],[-cage_max_dim,cage_max_dim],[0,max_height],[-max_velo, max_velo],[-max_velo, max_velo]])

    # Provide state to subgoal projection function.
    # a = np.concatenate((sim.data.qpos[:2], np.array([4 if sim.data.qvel[i] > 4 else -4 if sim.data.qvel[i] < -4 else sim.data.qvel[i] for i in range(3)])))
    project_state_to_subgoal = lambda sim, state: np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))


    # Set subgoal achievement thresholds
    velo_threshold = 0.8
    quat_threshold = 0.5
    # subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, quat_threshold, quat_threshold, quat_threshold, quat_threshold, velo_threshold, velo_threshold, velo_threshold])
    subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, velo_threshold, velo_threshold])


    # To properly visualize goals, update "display_end_goal" and "display_subgoals" methods in "environment.py"


    """
    3. SET MISCELLANEOUS HYPERPARAMETERS

    Below are some other agent hyperparameters that can affect results, including
        a. Subgoal testing percentage
        b. Subgoal penalty
        c. Exploration noise
        d. Replay buffer size
    """

    agent_params = {}

    # Define percentage of actions that a subgoal level (i.e. level i > 0) will test subgoal actions
    agent_params["subgoal_test_perc"] = 0.3

    # Define subgoal penalty for missing subgoal.  Please note that by default the Q value target for missed subgoals does not include Q-value of next state (i.e, discount rate = 0).  As a result, the Q-value target for missed subgoal just equals penalty.  For instance in this 3-level UR5 implementation, if a level proposes a subgoal and misses it, the Q target value for this action would be -10.  To incorporate the next state in the penalty, go to the "penalize_subgoal" method in the "layer.py" file.
    agent_params["subgoal_penalty"] = -FLAGS.time_scale


    # Set exploration hyperparameters

    # Set percentage of non-subgoal testing time, agents will take random actions (sampled uniformly from action space)
    agent_params["random_action_perc"] = 0.3


    # Define exploration noise that is added to both subgoal actions and atomic actions.  Noise added is Gaussian N(0, noise_percentage * action_dim_range)
    agent_params["atomic_noise"] = [0.2 for i in range(8)]
    agent_params["subgoal_noise"] = [0.2 for i in range(len(subgoal_thresholds))]

    """
    Set number of pre-learning episodes (i.e., number of initial episodes before any updates are made to actor and critic functions).

    We noticed that this buffer period was helpful for ensuring the policy chose actions that could be achieved within $H$ actions.
    This is likely because the agent is able to gather some subgoal penalty transitions.
    """
    agent_params["num_pre_training_episodes"] = 30

    # Define number of episodes of transitions to be stored by each level of the hierarchy
    agent_params["episodes_to_store"] = 500

    # Provide training schedule for agent.  Training by default will alternate between exploration and testing.  Hyperparameter below indicates number of exploration episodes.  Testing occurs for 100 episodes.  To change number of testing episodes, go to "ran_HAC.py".
    agent_params["num_exploration_episodes"] = 100

    # For other relavent agent hyperparameters, please refer to the "agent.py" and "layer.py" files



    # Ensure environment customization have been properly entered
    check_validity(model_name, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space, subgoal_bounds, subgoal_thresholds, max_actions, timesteps_per_action)


    # Instantiate and return agent and environment
    env = Environment(model_name, reset_fixed, reset_coordinates, uniform_goal_sampling, maze_array, tile_size, goal_space_train,
                      goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space,
                      subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions, timesteps_per_action,
                      FLAGS.show)

    agent = Agent(FLAGS,env,agent_params, simulation_id, outputs_directory)

    return agent, env
