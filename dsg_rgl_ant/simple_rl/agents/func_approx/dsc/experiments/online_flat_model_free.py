import os
import ipdb
import time
import torch
import pickle
import random
import argparse
import numpy as np
from copy import deepcopy
from collections import deque
from dsg_rgl_ant.simple_rl.agents.func_approx.dsc.experiments.utils import *
from dsg_rgl_ant.simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP

from dsg_rgl_ant.simple_rl.agents.func_approx.td3.TD3AgentClass import TD3


class OnlineModelBasedSkillChaining(object):
    def __init__(self, max_steps, stronger_baseline,
                 use_diverse_starts, use_dense_rewards, use_optimal_sampler, lr_c, lr_a,
                 experiment_name, device, logging_freq, evaluation_freq, seed):
        
        self.device = device
        self.experiment_name = experiment_name
        self.max_steps = max_steps
        self.use_diverse_starts = use_diverse_starts
        self.use_dense_rewards = use_dense_rewards
        self.use_optimal_sampler = use_optimal_sampler
        self.stronger_baseline = stronger_baseline

        self.seed = seed
        self.logging_freq = logging_freq
        self.evaluation_freq = evaluation_freq

        self.mdp = D4RLAntMazeMDP("umaze", goal_state=np.array((0, 8)), seed=seed)
        
        self.agent = TD3(state_dim=self.mdp.state_space_size()+2,
                                    action_dim=self.mdp.action_space_size(),
                                    max_action=1.,
                                    name="td3-agent",
                                    device=self.device,
                                    lr_c=lr_c, lr_a=lr_a,
                                    use_output_normalization=False)
        
        self.target_salient_event = self.mdp.get_original_target_events()[0]
        self.goal = self.target_salient_event.get_target_position()

        self.log = {}

    def act(self, aug_state, eval_mode=False):
        if random.random() < 0.1:
            return self.mdp.sample_random_action()
        return self.agent.act(aug_state, evaluation_mode=eval_mode)

    def get_augmented_state(self, state, goal):
        return np.concatenate((state.features(), goal))

    def experience_replay(self, trajectory, goal):
        for state, action, reward, next_state in trajectory:
            augmented_state = self.get_augmented_state(state, goal)
            augmented_next_state = self.get_augmented_state(next_state, goal)

            reward_func = self.mdp.dense_gc_reward_function if self.use_dense_rewards \
                else self.mdp.sparse_gc_reward_function
            reward, global_done = reward_func(next_state, goal, info={})

            self.agent.step(augmented_state, action, reward, augmented_next_state, global_done)

    def her_rollout(self, steps, eval_mode=False):        
        trajectory = []
        step_number = 0
        state = deepcopy(self.mdp.cur_state)

        while step_number < steps and not self.mdp.sparse_gc_reward_function(state, self.goal, {})[1]:
            if step_number % 200 == 0:
                print(f"[Step: {step_number}] Rolling out from {state.position} targeting {self.goal}")
            
            state = deepcopy(self.mdp.cur_state)
            aug_state = self.get_augmented_state(state, self.goal)
            action = self.act(aug_state, eval_mode=eval_mode)
            reward, next_state = self.mdp.execute_agent_action(action)

            reward_func = self.mdp.dense_gc_reward_function if self.use_dense_rewards \
                else self.mdp.sparse_gc_reward_function
            reward, done = reward_func(next_state, self.goal, info={})

            trajectory.append((state, action, reward, next_state))
            if done:
                break
            
            step_number += 1

        self.experience_replay(trajectory, self.goal)

        return step_number
    
    def run_loop(self, num_episodes, num_steps, start_episode=0):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(start_episode, start_episode + num_episodes):
            self.reset()
            
            if self.stronger_baseline:
                self.goal = self.mdp.get_position(self.mdp.sample_random_state())

            step = self.her_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            self.log_success_metrics(episode)

        return per_episode_durations

    def log_success_metrics(self, episode):
        if episode % self.evaluation_freq == 0:
            success, step_count = test_agent(self, 1, self.max_steps)

            self.log[episode] = {}
            self.log[episode]["success"] = success
            self.log[episode]["step-count"] = step_count[0]

            with open(f"{self.experiment_name}/log_file_{self.seed}.pkl", "wb+") as log_file:
                pickle.dump(self.log, log_file)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")

        if episode % self.logging_freq == 0 and episode != 0:
            self.goal = self.target_salient_event.get_target_position()
            make_chunked_goal_conditioned_value_function_plot(self.agent,
                                                            goal=self.goal,
                                                            episode=episode, seed=self.seed,
                                                            experiment_name=self.experiment_name)

    def reset(self):
        self.mdp.reset()

        if self.use_diverse_starts:
            random_state = self.mdp.sample_random_state()
            random_position = self.mdp.get_position(random_state)
            self.mdp.set_xy(random_position)


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

def test_agent(exp, num_experiments, num_steps):
    def rollout():
        step_number = 0
        while step_number < num_steps and not exp.mdp.sparse_gc_reward_function(exp.mdp.cur_state, exp.goal, {})[1]:
            state = deepcopy(exp.mdp.cur_state)
            aug_state = exp.get_augmented_state(state, exp.goal)
            action = exp.act(aug_state, eval_mode=True)
            exp.mdp.execute_agent_action(action)
            step_number += 1
        return step_number
        
    success = 0
    step_counts = []
    exp.goal = exp.target_salient_event.get_target_position()

    for _ in tqdm(range(num_experiments), desc="Performing test rollout"):
        exp.mdp.reset()
        steps_taken = rollout()
        if steps_taken != num_steps or exp.target_salient_event(exp.mdp.cur_state):
            success += 1
        step_counts.append(steps_taken)

    print("*" * 80)
    print(f"Test Rollout Success Rate: {success / num_experiments}, Duration: {np.mean(step_counts)}")
    print("*" * 80)

    return success / num_experiments, step_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--use_diverse_starts", action="store_true", default=False)
    parser.add_argument("--use_dense_rewards", action="store_true", default=False)
    parser.add_argument("--use_optimal_sampler", action="store_true", default=False)
    parser.add_argument("--logging_frequency", type=int, default=50, help="Draw init sets, etc after every _ episodes")
    parser.add_argument("--evaluation_frequency", type=int, default=10)
    parser.add_argument("--stronger_baseline", action="store_true", default=False)

    parser.add_argument("--lr_c", type=float, help="critic learning rate")
    parser.add_argument("--lr_a", type=float, help="actor learning rate")
    args = parser.parse_args()

    exp = OnlineModelBasedSkillChaining(experiment_name=args.experiment_name,
                                        device=torch.device(args.device),
                                        max_steps=args.steps,
                                        use_diverse_starts=args.use_diverse_starts,
                                        use_dense_rewards=args.use_dense_rewards,
                                        use_optimal_sampler=args.use_optimal_sampler,
                                        logging_freq=args.logging_frequency,
                                        evaluation_freq=args.evaluation_frequency,
                                        stronger_baseline=args.stronger_baseline,
                                        seed=args.seed,
                                        lr_c=args.lr_c,
                                        lr_a=args.lr_a)

    create_log_dir(args.experiment_name)
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    start_time = time.time()
    durations = exp.run_loop(args.episodes, args.steps)
    end_time = time.time()

    print("TIME: ", end_time - start_time)

