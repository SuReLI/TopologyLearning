import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from .hac_utils import layer


class Actor:

    def __init__(self,
            sess,
            env,
            layer_number):

        self.sess = sess
        self.layer_number = layer_number

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = env.action_space.high
            self.action_offset = np.zeros(8)  # TODO: remove action_offset
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = np.concatenate((env.maze_space.high, np.array([0.5, 3, 3])))
            self.action_offset = np.array([0., 0., 0.5, 0., 0.])

        # Dimensions of action will depend on layer level
        self.action_space_size = env.action_size if layer_number == 0 else 5
        self.actor_name = 'actor_' + str(layer_number)

        # Dimensions of goal placeholder will differ depending on layer level
        self.goal_dim = env.goal_size if layer_number == 2 else 5
        self.state_dim = env.observation_size

        self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))
        self.features_ph = tf.concat([self.state_ph, self.goal_ph], axis=1)

        # Create actor network
        self.infer = self.create_nn(self.features_ph)

    def get_action(self, state, goal):

        if self.layer_number == 0:
            if isinstance(state, list):
                state_ = [np.concatenate((np.zeros(len(g)), s[len(g):])) for s, g in zip(state, goal)]
                goal_ = [g - s[:len(g)] for s, g in zip(state, goal)]
            else:
                state_ = np.concatenate((np.zeros(goal.shape[-1]), state[-1, goal.shape[-1]:]))[np.newaxis]
                goal_ = (goal[-1] - state[-1, :goal.shape[-1]])[np.newaxis]
            state, goal = state_, goal_

        actions = self.sess.run(self.infer,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal
                })

        return actions

    # def create_nn(self, state, goal, name='actor'):
    def create_nn(self, features, name=None):

        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset

        return output
