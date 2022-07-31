
"""
This class exploit a goal-conditioned policy trained with HAC (https://arxiv.org/pdf/1712.00948.pdf).
This class is not made for training.
"""

# TODO: Implement get_qvalue
# TODO: supprimer toutes les fonctions d'entraînement

import numpy as np
from .layer import Layer
from ..environment import Environment
import pickle as cpickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import pickle as cpickle


# Below class instantiates an agent
class LowPolicyAgent:
    def __init__(self, FLAGS, env, agent_params):

        self.FLAGS = FLAGS
        self.sess = tf.Session()

        # Create agent with number of levels specified by user
        # NB: impossible to import networks weights without re-creating the full class, bu only the lowest level layer
        # will be exploited.
        self.layers = [Layer(i, FLAGS, env, self.sess, agent_params) for i in range(FLAGS.layers)]

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        # Initialize actor/critic networks. Load saved parameters
        self.initialize_networks()

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        # Below parameters will be used to store performance results
        self.performance_log = []

        self.other_params = agent_params

    # Determine whether each layer's goal was achieved.
    # Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self, env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.FLAGS.layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim, self.current_state)
        proj_end_goal = env.project_state_to_end_goal(env.sim, self.current_state)

        for i in range(self.FLAGS.layers):

            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.FLAGS.layers - 1:

                # Check dimensions are appropriate
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(env.end_goal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.subgoal_thresholds), \
                    "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved

    def initialize_networks(self):

        model_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(model_vars)

        # Set up directory for saving models
        self.model_dir = os.getcwd() + '/models'
        self.model_loc = self.model_dir + '/HAC.ckpt'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Initialize actor/critic networks
        self.sess.run(tf.global_variables_initializer())

        # If not retraining, restore weights
        # if we are not retraining from scratch, just restore weights
        if not self.FLAGS.retrain:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

    # Save neural network parameters
    def save_model(self, episode):
        self.saver.save(self.sess, self.model_loc, global_step=episode)

    def action(self, state, goal) -> np.ndarray:
        """
        Used by the global agent to get the low-level policy action.
        """
        actor = self.layers[0].actor
        state -= goal  # Compute s - g
        goal = np.zeros(goal.shape)
        return actor.get_action(state[np.newaxis], goal[np.newaxis])[0]

    # Save performance evaluations
    def log_performance(self, success_rate):

        # Add latest success_rate to list
        self.performance_log.append(success_rate)

        # Save log
        cpickle.dump(self.performance_log, open("../performance_log.p", "wb"))
