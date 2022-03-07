from random import choice
from statistics import mean

from agents.gc_agent import GoalConditionedAgent


def pre_train(goal_reaching_agent: GoalConditionedAgent, environment, nb_episodes=400, time_steps_max_per_episode=80):
    print("Pretraining low level agent ... please wait a bit ...")
    average_training_accuracy = []
    local_buffer = []
    goal_reaching_agent.on_simulation_start()
    for episode_id in range(nb_episodes):
        training_accuracy = []
        time_step_id = 0
        state = environment.reset()
        if episode_id != 0:
            goal = choice(local_buffer)
            goal_reaching_agent.on_episode_start(state, goal)

        episode_rewards = []
        done = False
        while (not done) and time_step_id < time_steps_max_per_episode:

            if episode_id != 0:
                action = goal_reaching_agent.action(state)
            else:
                action = environment.action_space.sample()
            state, reward, _, _ = environment.step(action)
            local_buffer.append(state)
            if episode_id != 0:
                if (state == goal).all():
                    done = True
                    reward = 1
                    training_accuracy.append(1)
                else:
                    reward = -1

            # Ending time step process ...
            if episode_id != 0:
                goal_reaching_agent.on_action_stop(action, state, reward, done)

            # Store reward
            episode_rewards.append(reward)
            time_step_id += 1
        if episode_id != 0:
            if not done:
                training_accuracy.append(0)
            average_training_accuracy.append(mean(training_accuracy))

            goal_reaching_agent.on_episode_stop()
        environment.close()

    if len(average_training_accuracy) > 20:
        last_20_average = mean(average_training_accuracy[-20:])
    else:
        last_20_average = mean(average_training_accuracy)
    return last_20_average
