"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""
import os.path

import numpy as np

from utils.sys_fun import save_image, generate_video

NUM_BATCH = 1000
TEST_FREQ = 2

num_test_episodes = 2000

def run_HAC(env, agent):

    # Determine training mode.
    # If not testing and not solely training, interleave training and testing to track progress

    # Track total training episodes completed
    total_episodes = 0
    reached_goals = []
    failed_goals = []

    # Evaluate policy every TEST_FREQ batches if interleaving training and testing
    print("\n--- TESTING ---")
    num_episodes = num_test_episodes

    # Reset successful episode counter
    successful_episodes = 0
    video = False

    for episode in range(num_episodes):

        print("Episode ", episode)
        episode_images = []

        # Train for an episode
        reached = False
        state, goal = env.reset()
        for interaction_id in range(500):
            action = agent.action(state, goal)
            state = env.step(action)
            if video:
                image = env.get_background_image()
                env.place_point(image, *state[:2], color=[0, 0, 255], width=10)
                env.place_point(image, *goal[:2], color=[0, 255, 0], width=10)
                episode_images.append(image)
            reached = ((state[:len(goal)] - goal) < env.goal_thresholds[:3]).all()
            if reached:
                successful_episodes += 1
                reached_goals.append(goal)
                break
        if not reached:
            if video:
                generate_video(episode_images, os.path.dirname(__file__), "test.mp4")
            failed_goals.append(goal)
        """
        if not reached:
            agent.save_episode_video()
            print()
        """

    agent.last_state = None
    print("#")

    # Log performance
    success_rate = successful_episodes / num_test_episodes * 100

    image = env.get_background_image()
    circle_width = int(image.shape[0] / (env.maze_space.high[0] * 2))

    env.place_point(image, 0, 0, [180, 180, 180], circle_width)
    for goal in reached_goals:
        env.place_point(image, *goal[:2], [0, 255, 0], 10)
    for goal in failed_goals:
        env.place_point(image, *goal[:2], [255, 0, 0], 10)
    dir = os.path.dirname(__file__)
    save_image(image, dir, "goals.png")

    print("\nTesting Success Rate %.2f%%" % success_rate)
    if success_rate > 70.:
        print()
    if success_rate > 90.:
        print()
    print("evaluation was made during ", num_test_episodes, ".")
    print("Success rate : ", success_rate, ".")

    print("\n--- END TESTING ---\n")
