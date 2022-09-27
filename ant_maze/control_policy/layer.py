
from .actor import Actor
from .critic import Critic


class Layer:
    def __init__(self, layer_number, env, sess):
        self.layer_number = layer_number
        self.sess = sess
        self.time_limit = 20

        self.current_state = None
        self.goal = None

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = 2

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit/3)

        # self.buffer_size = 10000000
        self.batch_size = 1024

        # Initialize actor and critic networks
        self.actor = Actor(sess, env, self.layer_number)
        self.critic = Critic(sess, env, self.layer_number)
