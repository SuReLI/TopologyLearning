"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""
import os.path
import pickle
import pickle as cpickle
import agent as Agent
from utils import print_summary
from math import ceil
import discord
from discord import SyncWebhook


TEST_FREQ = 2


def run_HAC(FLAGS, env, agent, simulation_id, outputs_directory, nb_interactions_before_evaluation, min_training_interactions,
            num_test_episodes):
    # For easier plotting, keep in memory at which time we made each evals.

    # Determine training mode.
    # If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not FLAGS.test and not FLAGS.train_only:
        mix_train_test = True

    evaluations_time_steps = []
    total_training_episodes = 0
    total_training_interactions = 0

    while total_training_interactions < min_training_interactions:

        num_episodes = 1

        # Should we do an evaluation
        if (mix_train_test and
                total_training_interactions >= nb_interactions_before_evaluation * len(evaluations_time_steps)):
            print("\n--- TESTING ---")
            agent.FLAGS.test = True
            evaluations_time_steps.append(total_training_interactions)
            num_episodes = num_test_episodes

            # Reset successful episode counter
            successful_episodes = 0

        for episode in range(num_episodes):

            # Train for an episode
            success, duration = agent.train(env, episode, total_training_episodes)
            if not agent.FLAGS.test:
                print("Episode ", total_training_episodes, ", end goal " + ("achieved" if success else "failed"), sep="")
                total_training_episodes += 1
                total_training_interactions += duration
            else:
                print("Eval episode ", episode, ", end goal " + ("achieved" if success else "failed"), sep="")
                if success:
                    successful_episodes += 1

        # Save agent
        agent.save_model(episode)

        # Finish evaluating policy if tested prior batch
        if mix_train_test and agent.FLAGS.test:

            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            agent.log_performance(success_rate)
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")

        with open(outputs_directory + "accuracies.pkl", "wb") as f:
            pickle.dump(agent.performance_log, f)
        with open(outputs_directory + "evaluations_time_steps.pkl", "wb") as f:
            pickle.dump(evaluations_time_steps, f)

    webhook_link = "https://discordapp.com/api/webhooks/1040647428333895821/" \
                   "MOrQET25fMfR3l1V9bUcjp-JTnr8HVHDbl5TEpbSirVhYqfdTHg0LayiLo-F5faDdgDK"
    send_discord_message("HAC test with name " + env.name.split(".")[0] + " with seed id " +
                         str(simulation_id) + " finished.", webhook_link=webhook_link)


class EmptyWebhookLinkError(ValueError):
    pass


def send_discord_message(message, webhook_link=os.environ.get("WEBHOOK_LINK", "")):
    if webhook_link == "":
        raise EmptyWebhookLinkError("You should set-up WEBHOOK_LINK environment variable to be able to send messages "
                                    "on the discord webhook. For more information: "
                                    "https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks")

    webhook = SyncWebhook.from_url(webhook_link)
    webhook.send(message)
