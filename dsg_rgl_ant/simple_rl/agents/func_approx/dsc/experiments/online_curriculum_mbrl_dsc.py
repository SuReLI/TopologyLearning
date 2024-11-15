import os
import ipdb
import torch
import pickle
import random
import argparse
import itertools
import numpy as np
from copy import deepcopy
from collections import deque
from dsg_rgl_ant.simple_rl.agents.func_approx.dsc.experiments.utils import *
from dsg_rgl_ant.simple_rl.agents.func_approx.dsc.MBOptionClass import ModelBasedOption
from dsg_rgl_ant.simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP


class OnlineModelBasedSkillChaining(object):
    def __init__(self, warmup_episodes, gestation_period, initiation_period, experiment_name, device):
        self.device = device
        self.experiment_name = experiment_name
        self.warmup_episodes = warmup_episodes

        self.gestation_period = gestation_period
        self.initiation_period = initiation_period

        self.mdp = D4RLAntMazeMDP("medium", goal_state=np.array((20, 20)))
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.global_option = self.create_global_model_based_option()
        self.goal_option = self.create_model_based_option(name="goal-option", parent=None)

        self.chain = [self.goal_option]
        self.new_options = [self.goal_option]  # options in their gestation phase
        self.mature_options = []               # options in their initiation and initiation-done phases

    def act(self, state):
        learning_options = [o for o in self.mature_options + self.new_options if o.get_training_phase() != "initiation_done"]
        if len(learning_options) > 0:
            assert len(learning_options) == 1
            if learning_options[0].is_init_true(state):
                return learning_options[0]
            
        for option in self.chain:
            if option.is_init_true(state) and not option.is_term_true(state):
                return option
        return self.global_option

    def random_rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_state.is_terminal():
            state = deepcopy(self.mdp.cur_state)
            action = self.mdp.sample_random_action()
            reward, next_state = self.mdp.execute_agent_action(action)
            self.global_option.update(state, action, reward, next_state)
            step_number += 1
        return step_number

    def dsc_rollout(self, num_steps):
        step_number = 0
        while step_number < num_steps and not self.mdp.cur_state.is_terminal():
            state = deepcopy(self.mdp.cur_state)
            selected_option = self.act(state)

            transitions, reward = selected_option.rollout(step_number=step_number)
            self.manage_chain_after_rollout(selected_option)

            step_number += len(transitions)
        return step_number

    def run_loop(self, num_episodes=300, num_steps=150):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(num_episodes):
            self.reset(init=episode < self.warmup_episodes)

            step = self.dsc_rollout(num_steps) if episode > self.warmup_episodes else self.random_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            if episode >= self.warmup_episodes:
                self.learn_dynamics_model()

        return per_episode_durations

    def learn_dynamics_model(self):
        self.global_option.solver.load_data()
        self.global_option.solver.train(epochs=100, batch_size=1024)
        for option in self.chain:
            option.solver.model = self.global_option.solver.model

    def should_create_new_option(self):  # TODO: Cleanup
        if len(self.mature_options) > 0 and len(self.new_options) == 0:
            return self.mature_options[-1].get_training_phase() == "initiation_done" and \
                not self.mature_options[-1].pessimistic_is_init_true(self.mdp.init_state)
        return False

    def manage_chain_after_rollout(self, executed_option):

        if executed_option in self.new_options and executed_option.get_training_phase() != "gestation":
            self.new_options.remove(executed_option)
            self.mature_options.append(executed_option)

        if self.should_create_new_option():
            name = f"option-{len(self.mature_options)}"
            new_option = self.create_model_based_option(name, parent=self.mature_options[-1])
            print(f"Creating {name} with parent {new_option.parent}, new_options = {self.new_options}, mature_options = {self.mature_options}")
            self.new_options.append(new_option)
            self.chain.append(new_option)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")
        options = [o for o in self.mature_options + self.new_options if o.get_training_phase() != "gestation"]
        for option in options:
            plot_two_class_classifier(option, episode, self.experiment_name, plot_examples=True)

    def create_model_based_option(self, name, parent=None):
        option = ModelBasedOption(parent=parent, mdp=self.mdp,
                                  buffer_length=50, global_init=False,
                                  gestation_period=self.gestation_period,
                                  initiation_period=self.initiation_period,
                                  timeout=200, max_steps=1000, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name=name,
                                  path_to_model="",
                                  global_solver=self.global_option.solver)
        return option

    def create_global_model_based_option(self):  # TODO: what should the timeout be for this option?
                                                 # TODO: Should we pick sub-goals for the global-option?
        option = ModelBasedOption(parent=None, mdp=self.mdp,
                                  buffer_length=50, global_init=True,
                                  gestation_period=self.gestation_period,
                                  initiation_period=self.initiation_period,
                                  timeout=100, max_steps=1000, device=self.device,
                                  target_salient_event=self.target_salient_event,
                                  name="global-option",
                                  path_to_model="",
                                  global_solver=None)
        return option

    def reset(self, init):
        if init:
            self.mdp.reset()
            return

        gestation_options = self.new_options
        init_options = [option for option in self.mature_options if option.get_training_phase() == "initiation"]

        if len(init_options) > 0:
            assert len(init_options) == 1, init_options
            assert len(init_options[0].positive_examples) > 0

            print(f"[Initiation-Reset] Resetting to {init_options[0]}'s initiation region")
            self.reset_to_option_positive_example(init_options[0])

        elif len(gestation_options) > 0:
            assert len(gestation_options) == 1, gestation_options
            
            if gestation_options[0].parent is None:
                print(f"[Gestation-Reset-1] Resetting to {gestation_options[0]}'s initiation region")
                self.reset_to_satisfy_condition(condition=lambda s: s[0] > 15 and s[1] > 10 and not self.mdp.is_goal_state(s))

            # if we have some positive examples, better to sample from them
            elif len(gestation_options[0].positive_examples) > 0:
                option = gestation_options[0]  # type: ModelBasedOption

                print(f"[Gestation-Reset-2] Resetting to {gestation_options[0]}'s initiation region")
                self.reset_to_option_positive_example(option)

            # if we don't have any positive examples, sample close-ish to a parent positive
            elif gestation_options[0].parent.pessimistic_classifier is not None:
                option = gestation_options[0]  # type: ModelBasedOption
                
                goal = option.parent.sample_from_initiation_region()
                print(f"[Gestation-Reset-3] Resetting to {gestation_options[0]}'s initiation region")
                self.reset_to_satisfy_condition(condition=lambda s: np.linalg.norm(self.mdp.get_position(s) - goal) <= 10 and not self.mdp.is_goal_state(s))
        else:
            self.mdp.reset()

    def reset_to_option_positive_example(self, option):
        positive_examples = list(itertools.chain.from_iterable(option.positive_examples))
        sampled_positive = None

        while sampled_positive is None:
            sampled_positive = random.choice(positive_examples)
            if not option.is_init_true(sampled_positive) or option.is_term_true(sampled_positive):
                sampled_positive = None

        self.mdp.reset()
        self.mdp.set_xy(sampled_positive)

    def reset_to_satisfy_condition(self, condition):
        """ Reset the simulator inside the optimistic classifier of the input option. """

        start_state = None
        num_tries = 0
        while start_state is None and num_tries < 50000:
            num_tries += 1
            s = self.mdp.sample_random_state()
            start_state = s if condition(s) else None

        self.mdp.reset()

        start_position = self.mdp.get_position(start_state)
        if start_position is not None:
            print(f"Resetting state to {start_position}")
            self.mdp.set_xy(start_position)

    def save_option_dynamics_data(self):
        """ Save data that can be used to learn a model of the option dynamics. """
        for option in self.mature_options:
            states = np.array([pair[0] for pair in option.in_out_pairs])
            next_states = np.array([pair[1] for pair in option.in_out_pairs])
            with open(f"{self.experiment_name}/{option.name}-dynamics-data.pkl", "wb+") as f:
                pickle.dump((states, next_states), f)


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--gestation_period", type=int, default=3)
    parser.add_argument("--initiation_period", type=float, default=3)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save_option_data", action="store_true", default=False)
    parser.add_argument("--warmup_episodes", type=int, default=5)
    args = parser.parse_args()

    exp = OnlineModelBasedSkillChaining(gestation_period=args.gestation_period,
                                        initiation_period=args.initiation_period,
                                        experiment_name=args.experiment_name,
                                        device=torch.device(args.device),
                                        warmup_episodes=args.warmup_episodes)

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    durations = exp.run_loop(args.episodes, args.steps)

    if args.save_option_data:
        exp.save_option_dynamics_data()
