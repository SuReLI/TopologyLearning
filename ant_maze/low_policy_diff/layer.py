import numpy as np
from .experience_buffer import ExperienceBuffer
from .actor import Actor
from .critic import Critic
from time import sleep


class Layer:
    def __init__(self, layer_number, FLAGS, env, sess):
        self.layer_number = layer_number
        self.FLAGS = FLAGS
        self.sess = sess

        # Initialize actor and critic networks
        self.actor = Actor(sess, env, self.layer_number, FLAGS)
        self.critic = Critic(sess, env, self.layer_number, FLAGS)
