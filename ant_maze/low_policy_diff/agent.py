import numpy as np
from .layer import Layer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os


# Below class instantiates an agent
class LowPolicyAgentDiff:
    def __init__(self,FLAGS, env, agent_params):

        self.FLAGS = FLAGS
        self.sess = tf.Session()

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        # Create agent with number of levels specified by user
        self.layers = [Layer(i, FLAGS, env, self.sess) for i in range(FLAGS.layers)]

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_loc = None

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        self.initialize_networks()

    def action(self, state, goal):
        """
        Exploit the loaded policy to choose an action with the layer 0.
        """
        _state = state[goal.shape[0]] if len(state.shape) == 1 else state[:, goal.shape[-1]]
        _state_goal = _state - goal
        _goal = np.zeros(goal.shape)
        return self.layers[0].actor.get_action(_state_goal[np.newaxis], _goal[np.newaxis])[0]

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
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
