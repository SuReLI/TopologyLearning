# Python imports.
from __future__ import print_function
import random
from collections import defaultdict

# Other imports.
from dsg_rgl_ant.simple_rl.abstraction.action_abs.PolicyClass import Policy

class PolicyFromDict(Policy):

    def __init__(self, policy_dict={}):
        self.policy_dict = policy_dict

    def get_action(self, state):
        if state not in self.policy_dict.keys():
            print("(PolicyFromDict) Warning: unseen state (" + str(state) + "). Acting randomly.")
            return random.choice(list(set(self.policy_dict.values())))
        else:
            return self.policy_dict[state]
