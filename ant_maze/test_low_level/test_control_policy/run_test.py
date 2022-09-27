"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""
import pickle

import numpy as np
from matplotlib import pyplot as plt
from ant_maze.control_policy.agent import AntMazeControlPolicy
from ant_maze.environment import AntMaze

num_episodes = 500

env = AntMaze(show=False)
agent = AntMazeControlPolicy(env)

successful_episodes = 0
reached_goals = []
failed_goals = []
for episode in range(num_episodes):
    state, goal = env.reset()
    reached = False
    for i in range(700):
        # action = agent.layers[0].actor.get_action(state[np.newaxis], goal[np.newaxis])[0]
        action = agent.action(state, goal)
        # state = env.execute_action(action)
        state = env.step(action)
        reached = (np.absolute(goal - state[:len(goal)]) < np.array([0.5, 0.5, 0.8, 0.5, 0.5])[:len(goal)]).all()
        # print("step ", i, ", distance ", np.absolute(goal - state[:len(goal)]))

    if reached:
        reached_goals.append(goal[:2])
    else:
        failed_goals.append(goal[:2])

    if reached:
        print("Episode ", episode, ", succeed")
        successful_episodes += 1
    else:
        print("Episode ", episode, ", failed")
    print("Running success rate: ", successful_episodes / (episode + 1) * 100)

with open('reached_sub_goals.pkl', 'wb') as f:
    pickle.dump(reached_goals, f)

reached_goals = np.array(reached_goals)
failed_goals = np.array(failed_goals)
plt.scatter(reached_goals[:, 0], reached_goals[:, 1], c="g")
plt.scatter(failed_goals[:, 0], failed_goals[:, 1], c="r")
plt.scatter(0., 0., c="#b3b3b3")
plt.show()

# Finish evaluating policy if tested prior batch

# Log performance
success_rate = successful_episodes / num_episodes * 100
print("\nTesting Success Rate %.2f%%" % success_rate)