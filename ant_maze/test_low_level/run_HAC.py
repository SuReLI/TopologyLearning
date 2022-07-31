"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""

NUM_BATCH = 1000
TEST_FREQ = 2

num_test_episodes = 100


def run_HAC(env, agent):

    # Determine training mode.
    # If not testing and not solely training, interleave training and testing to track progress

    # Track total training episodes completed
    total_episodes = 0
    for batch in range(NUM_BATCH):
        test = False
        num_episodes = agent.other_params["num_exploration_episodes"]

        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        print("\n--- TESTING ---")
        num_episodes = num_test_episodes

        # Reset successful episode counter
        successful_episodes = 0

        for episode in range(num_episodes):

            print("\nBatch %d, Episode %d" % (batch, episode))

            # Train for an episode

            goal = env.get_next_goal(test=True)
            state = env.reset_sim(goal)
            for interaction_id in range(500):
                action = agent.action(state, goal)
                next_state = env.execute_action(action)
                if ((next_state - goal) < env.end_goal_thresholds).all():
                    successful_episodes += 1
                    break
        print("#")

        # Log performance
        success_rate = successful_episodes / num_test_episodes * 100
        print("\nTesting Success Rate %.2f%%" % success_rate)
        if success_rate > 70.:
            print(agent.performance_log)
        if success_rate > 90.:
            print()
        agent.log_performance(success_rate)
        agent.FLAGS.test = False

        print("\n--- END TESTING ---\n")
