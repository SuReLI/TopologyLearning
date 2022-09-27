import numpy as np
from .layer import Layer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os


# Below class instantiates an agent
class AntMazeControlPolicy:
    def __init__(self, env):

        self.sess = tf.Session()
        self.nb_layers = 3
        self.h = 20

        # Create agent with number of levels specified by user
        self.layers = [Layer(i, env, self.sess) for i in range(self.nb_layers)]

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        self.initialize_networks()

    def action(self, state, goal):
        goal_ = np.concatenate((goal, state[len(goal):5]))
        action = self.layers[0].actor.get_action(state[np.newaxis], goal_[np.newaxis])[0]
        """
        sub_goal_1 = self.layers[2].actor.get_action(np.reshape(state, (1, len(state))),
                                                     np.reshape(goal, (1, len(goal))))[0]

        sub_goal_0 = self.layers[1].actor.get_action(np.reshape(state, (1, len(state))),
                                                     np.reshape(sub_goal_1, (1, len(sub_goal_1))))[0]
        action = self.layers[0].actor.get_action(np.reshape(state, (1, len(state))),
                                               np.reshape(sub_goal_0, (1, len(sub_goal_0))))[0]
        """

        return action

    def get_q_value(self, state, goal):
        goal_ = np.concatenate((goal, state[len(goal):5]))
        action = self.action(state, goal_)
        state_ = np.concatenate((np.zeros(len(goal)), state[len(goal):]))
        goal_ = np.concatenate((goal - state[:len(goal)], state[len(goal):3], state[3:5]))
        q_value = self.layers[0].critic.get_Q_value(state_[np.newaxis], goal_[np.newaxis], action[np.newaxis])
        return q_value

    def initialize_networks(self):

        model_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(model_vars)

        # Set up directory for saving models
        self.model_dir = os.path.dirname(__file__) + '/models'
        self.model_loc = self.model_dir + '/HAC.ckpt'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

         # Initialize actor/critic networks
        self.sess.run(tf.global_variables_initializer())

        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
