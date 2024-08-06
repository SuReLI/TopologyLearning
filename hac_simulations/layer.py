import numpy as np
from experience_buffer import ExperienceBuffer
from actor import Actor
from critic import Critic
from time import sleep

class Layer():
    def __init__(self, layer_number, FLAGS, env, sess, agent_params):
        self.layer_number = layer_number
        self.FLAGS = FLAGS
        self.sess = sess

        # Set time limit for each layer.  If agent uses only 1 layer, time limit is the max number of low-level actions allowed in the episode (i.e, env.max_actions).
        if FLAGS.layers > 1:
            self.time_limit = FLAGS.time_scale
        else:
            self.time_limit = env.max_actions

        self.current_state = None
        self.goal = None

        # Initialize Replay Buffer.  Below variables determine size of replay buffer.

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**7

        # Number of full episodes stored in replay buffer
        self.episodes_to_store = agent_params["episodes_to_store"]

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = 2

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit/3)

        # Buffer size = transitions per attempt * # attempts per episode * num of episodes stored
        self.buffer_size = min(self.trans_per_attempt * self.time_limit**(self.FLAGS.layers-1 - self.layer_number) * self.episodes_to_store, self.buffer_size_ceiling)

        # self.buffer_size = 10000000
        self.batch_size = 1024
        self.replay_buffer = ExperienceBuffer(self.buffer_size, self.batch_size)

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        # Initialize actor and critic networks
        self.actor = Actor(sess, env, self.batch_size, self.layer_number, FLAGS)
        self.critic = Critic(sess, env, self.layer_number, FLAGS)

        # Parameter determines degree of noise added to actions during training
        # self.noise_perc = noise_perc
        if self.layer_number == 0:
            self.noise_perc = agent_params["atomic_noise"]
        else:
            self.noise_perc = agent_params["subgoal_noise"]

        # Create flag to indicate when layer has ran out of attempts to achieve goal.  This will be important for subgoal testing
        self.maxed_out = False

        self.subgoal_penalty = agent_params["subgoal_penalty"]



    # Add noise to provided action
    def add_noise(self,action, env):

        # Noise added will be percentage of range
        if self.layer_number == 0:
            action_bounds = env.action_bounds
            action_offset = env.action_offset
        else:
            action_bounds = env.subgoal_bounds_symmetric
            action_offset = env.subgoal_bounds_offset

        assert len(action) == len(action_bounds), "Action bounds must have same dimension as action"
        assert len(action) == len(self.noise_perc), "Noise percentage vector must have same dimension as action"

        # Add noise to action and ensure remains within bounds
        for i in range(len(action)):
            action[i] += np.random.normal(0,self.noise_perc[i] * action_bounds[i])

            action[i] = max(min(action[i], action_bounds[i]+action_offset[i]), -action_bounds[i]+action_offset[i])

        return action


    # Select random action
    def get_random_action(self, env):

        if self.layer_number == 0:
            action = np.zeros((env.action_dim))
        else:
            action = np.zeros((env.subgoal_dim))

        # Each dimension of random action should take some value in the dimension's range
        for i in range(len(action)):
            if self.layer_number == 0:
                action[i] = np.random.uniform(-env.action_bounds[i] + env.action_offset[i], env.action_bounds[i] + env.action_offset[i])
            else:
                action[i] = np.random.uniform(env.subgoal_bounds[i][0],env.subgoal_bounds[i][1])

        return action


    # Function selects action using an epsilon-greedy policy
    def choose_action(self,agent, env, subgoal_test):

        # If testing mode or testing subgoals, action is output of actor network without noise
        if agent.FLAGS.test or subgoal_test:

            return self.actor.get_action(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))))[0], "Policy", subgoal_test

        else:

            if np.random.random_sample() > agent.other_params["random_action_perc"]:
                # Choose noisy action
                action = self.add_noise(self.actor.get_action(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))))[0],env)

                action_type = "Noisy Policy"

            # Otherwise, choose random action
            else:
                action = self.get_random_action(env)

                action_type = "Random"

            # Determine whether to test upcoming subgoal
            if np.random.random_sample() < agent.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False



            return action, action_type, next_subgoal_test


    # Create action replay transition by evaluating hindsight action given original goal
    def perform_action_replay(self, hindsight_action, next_state, goal_status):

        # Determine reward (0 if goal achieved, -1 otherwise) and finished boolean.  The finished boolean is used for determining the target for Q-value updates
        if goal_status[self.layer_number]:
            reward = 0
            finished = True
        else:
            reward = -1
            finished = False

        # Transition will take the form [old state, hindsight_action, reward, next_state, goal, terminate boolean, None]
        transition = [self.current_state.copy(), hindsight_action.copy(), reward, next_state.copy(), self.goal.copy(), finished, None]
        assert (- np.array([8., 8.]) <= hindsight_action.copy()[:2]).all() and \
               (hindsight_action.copy()[:2] <= np.array([8., 8.])).all()
        assert (- np.array([8., 8.]) <= self.goal.copy()[:2]).all() and \
               (self.goal.copy()[:2] <= np.array([8., 8.])).all()

        # Add action replay transition to layer's replay buffer
        self.replay_buffer.add(transition)


    # Create initial goal replay transitions
    def create_prelim_goal_replay_trans(self, hindsight_action, next_state, env, total_layers):

        # Create transition evaluating hindsight action for some goal to be determined in future.  Goal will be ultimately be selected from states layer has traversed through.  Transition will be in the form [old state, hindsight action, reward = None, next state, goal = None, finished = None, next state projeted to subgoal/end goal space]

        if self.layer_number == total_layers - 1:
            hindsight_goal = env.project_state_to_end_goal(env.sim, next_state)
            hindsight_goal[:2] = next_state[:2]
        else:
            hindsight_goal = env.project_state_to_subgoal(env.sim, next_state)
            hindsight_goal[:2] = next_state[:2]

        transition = [self.current_state.copy(), hindsight_action.copy(), None, next_state.copy(), None, None, hindsight_goal.copy()]

        assert (- np.array([8., 8.]) <= hindsight_action.copy()[:2]).all() and \
               (hindsight_action.copy()[:2] <= np.array([8., 8.])).all()
        assert (- np.array([8., 8.]) <= hindsight_goal.copy()[:2]).all() and \
               (hindsight_goal.copy()[:2] <= np.array([8., 8.])).all()
        self.temp_goal_replay_storage.append(transition)


    # Return reward given provided goal and goal achieved in hindsight
    def get_reward(self,new_goal, hindsight_goal, goal_thresholds):

        assert len(new_goal) == len(hindsight_goal) == len(goal_thresholds), "Goal, hindsight goal, and goal thresholds do not have same dimensions"

        # If the difference in any dimension is greater than threshold, goal not achieved
        for i in range(len(new_goal)):
            if np.absolute(new_goal[i]-hindsight_goal[i]) > goal_thresholds[i]:
                return -1

        # Else goal is achieved
        return 0

    # Finalize goal replay by filling in goal, reward, and finished boolean for the preliminary goal replay transitions created before
    def finalize_goal_replay(self,goal_thresholds):

        # Choose transitions to serve as goals during goal replay.  The last transition will always be used
        num_trans = len(self.temp_goal_replay_storage)

        num_replay_goals = self.num_replay_goals
        # If fewer transitions that ordinary number of replay goals, lower number of replay goals
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans

        indices = np.zeros((num_replay_goals))
        indices[:num_replay_goals-1] = np.random.randint(num_trans,size=num_replay_goals-1)
        indices[num_replay_goals-1] = num_trans - 1
        indices = np.sort(indices)

        # For each selected transition, update the goal dimension of the selected transition and all prior transitions by using the next state of the selected transition as the new goal.  Given new goal, update the reward and finished boolean as well.
        for i in range(len(indices)):

            trans_copy = []
            for transition in self.temp_goal_replay_storage:
                t = []
                for elt in transition:
                    t.append(elt.copy() if isinstance(elt, np.ndarray) else elt)
                trans_copy.append(t)

            new_goal = trans_copy[int(indices[i])][6]
            # for index in range(int(indices[i])+1):
            for index in range(num_trans):
                # Update goal to new goal
                trans_copy[index][4] = new_goal

                # Update reward
                trans_copy[index][2] = self.get_reward(new_goal, trans_copy[index][6], goal_thresholds)

                # Update finished boolean based on reward
                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False

                assert (- np.array([8., 8.]) <= trans_copy[index][1].copy()[:2]).all() and (
                            trans_copy[index][1].copy()[:2] <= np.array([8., 8.])).all()
                assert (- np.array([8., 8.]) <= trans_copy[index][6].copy()[:2]).all() and (
                            trans_copy[index][6].copy()[:2] <= np.array([8., 8.])).all()
                self.replay_buffer.add(trans_copy[index])


        # Clear storage for preliminary goal replay transitions at end of goal replay
        self.temp_goal_replay_storage = []


    # Create transition penalizing subgoal if necessary.  The target Q-value when this transition is used will ignore next state as the finished boolena = True.  Change the finished boolean to False, if you would like the subgoal penalty to depend on the next state.
    def penalize_subgoal(self, subgoal, next_state, high_level_goal_achieved):
        transition = [self.current_state.copy(), subgoal.copy(), self.subgoal_penalty, next_state.copy(), self.goal.copy(), True, None]
        assert (- np.array([8., 8.]) <= subgoal.copy()[:2]).all() and \
               (subgoal.copy()[:2] <= np.array([8., 8.])).all()
        assert (- np.array([8., 8.]) <= self.goal.copy()[:2]).all() and \
               (self.goal.copy()[:2] <= np.array([8., 8.])).all()
        self.replay_buffer.add(transition)

    # Determine whether layer is finished training
    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached, (ii) maxed out episode time steps (env.max_actions), (iii) not testing and layer is out of attempts, and (iv) testing, layer is not the highest level, and layer is out of attempts.  NOTE: during testing, highest level will continue to ouput subgoals until either (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.

        # Return to previous level when any higher level goal achieved.  NOTE: if not testing and agent achieves end goal, training will continue until out of time (i.e., out of time steps or highest level runs out of attempts).  This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True

        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.FLAGS.test and self.layer_number < agent.FLAGS.layers-1 and attempts_made >= self.time_limit:
            return True

        else:
            return False


    # Learn to achieve goals with actions belonging to appropriate time scale.  "goal_array" contains the goal states for the current layer and all higher layers
    def train(self, agent, env, subgoal_test = False, episode_num = None):

        nb_interactions = 0

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state

        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        if self.layer_number == 0 and agent.FLAGS.show and agent.FLAGS.layers > 1:
            env.display_subgoals(agent.goal_array)

        # Current layer has self.time_limit attempts to each its goal state.
        attempts_made = 0

        while True:

            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, action_type, next_subgoal_test = self.choose_action(agent, env, subgoal_test)

            if self.FLAGS.Q_values:
                if self.layer_number == 2:
                    test_action = np.copy(action)
                    test_action[:3] = self.goal

            # If next layer is not bottom level, propose subgoal for next layer to achieve and determine whether that subgoal should be tested
            if self.layer_number > 0:

                agent.goal_array[self.layer_number - 1] = action
                for goal in agent.goal_array:
                    if goal is None:
                        continue
                    assert (- self.actor.action_space_bounds[:2] <= goal[:2]).all() and \
                           (goal[:2] <= self.actor.action_space_bounds[:2]).all()

                goal_status, max_lay_achieved, duration = agent.layers[self.layer_number - 1].train(agent, env, next_subgoal_test, episode_num)
                nb_interactions += duration

            # If layer is bottom level, execute low-level action
            else:
                next_state = env.execute_action(action)

                # Increment steps taken
                agent.steps_taken += 1

                agent.current_state = next_state

                # Determine whether any of the goals from any layer was achieved and, if applicable, the highest layer whose goal was achieved
                goal_status, max_lay_achieved = agent.check_goals(env)

                nb_interactions += 1

            attempts_made += 1

            # Perform hindsight learning using action actually executed (low-level action or hindsight subgoal)
            if self.layer_number == 0:
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as hindsight action
                if goal_status[self.layer_number-1]:
                    hindsight_action = action
                # Otherwise, use subgoal that was achieved in hindsight
                else:
                    hindsight_action = env.project_state_to_subgoal(env.sim, agent.current_state)
                    hindsight_action[:2] = agent.current_state[:2]


            # Next, create hindsight transitions if not testing
            if not agent.FLAGS.test:

                # Create action replay transition by evaluating hindsight action given current goal
                self.perform_action_replay(hindsight_action, agent.current_state, goal_status)

                # Create preliminary goal replay transitions.  The goal and reward in these transitions will be finalized when this layer has run out of attempts or the goal has been achieved.
                self.create_prelim_goal_replay_trans(hindsight_action, agent.current_state, env, agent.FLAGS.layers)

                # Penalize subgoals if subgoal testing and subgoal was missed by lower layers after maximum number of attempts
                if self.layer_number > 0 and next_subgoal_test and agent.layers[self.layer_number-1].maxed_out:
                    self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])

            # Update state of current layer
            self.current_state = agent.current_state

            # Return to previous level to receive next subgoal if applicable
            # if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or attempts_made >= self.time_limit:

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True

                # If not testing, finish goal replay by filling in missing goal and reward values before returning to prior level.
                if not agent.FLAGS.test:
                    if self.layer_number == agent.FLAGS.layers - 1:
                        goal_thresholds = env.end_goal_thresholds
                    else:
                        goal_thresholds = env.subgoal_thresholds

                    self.finalize_goal_replay(goal_thresholds)

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
                    return goal_status, max_lay_achieved, nb_interactions

    # Update actor and critic networks
    def learn(self, num_updates):

        for _ in range(num_updates):
            # Update weights of non-target networks
            # if self.replay_buffer.size >= self.batch_size:
            if self.replay_buffer.size > 250:
                old_states, actions, rewards, new_states, goals, is_terminals = self.replay_buffer.get_batch()
                goal_bounds = np.array([8., 8.])
                for goal in goals:
                    assert (- goal_bounds <= goal[:2]).all() and (goal[:2] <= goal_bounds).all()

                for state in new_states:
                    assert (- goal_bounds <= state[:2]).all() and (state[:2] <= goal_bounds).all()

                for state in old_states:
                    assert (- goal_bounds <= state[:2]).all() and (state[:2] <= goal_bounds).all()

                if self.layer_number > 0:
                    for action in actions:
                        assert (- goal_bounds <= action[:2]).all() and (action[:2] <= goal_bounds).all()

                next_batch_size = min(self.replay_buffer.size, self.replay_buffer.batch_size)


                self.critic.update(old_states, actions, rewards, new_states, goals, self.actor.get_action(new_states,goals), is_terminals)
                action_derivs = self.critic.get_gradients(old_states, goals, self.actor.get_action(old_states, goals))
                self.actor.update(old_states, goals, action_derivs, next_batch_size)
