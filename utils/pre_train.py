from random import choice

from old.src.agents.grid_world.graph_planning.sorb_oracle import SORB
from old.src.agents.grid_world.graph_planning.stc import STC_TL
from old.src.settings import settings


def pre_train(agent, environment):
    print("Pre-training agent ...")
    reached_goals = []
    _agent = agent if isinstance(agent, SORB) else agent.goal_reaching_agent
    _agent.on_simulation_start()
    local_buffer = []
    if isinstance(agent, STC_TL):
        pre_train_duration = settings.nb_episodes_per_simulation * 2
        # '-> Because we need to train the TC-Network once the policy is trained, for better quality training samples.
    else:
        pre_train_duration = settings.nb_episodes_per_simulation

    # Randomly explore environment to sample future goals
    for exploration_id in range(settings.nb_pretrain_initial_random_explorations):
        state, _ = environment.reset()
        for exploration_step_id in range(settings.pre_train_initial_random_exploration_duration):
            action = environment.action_space.sample()
            state, _, _ = environment.learning_step(action)
            local_buffer.append(state)

    # Used observed states to train the policy
    for episode_id in range(pre_train_duration):
        time_step_id = 0
        state, _ = environment.reset()
        if isinstance(agent, STC_TL):
            # Store samples for TC-Network training
            agent.last_episode_trajectory = [state]
        goal = choice(local_buffer)
        _agent.on_episode_start(state, goal)
        done = False
        while (not done) and time_step_id < settings.episode_length:
            if episode_id != 0:
                action = _agent.action(state)
            else:
                action = environment.action_space.sample()
            state, reward, _ = environment.learning_step(action)
            if isinstance(agent, STC_TL):
                # Train TC-Network
                agent.last_episode_trajectory.append(state)
                agent.train_tc_network()
            local_buffer.append(state)
            if (state == goal).all():
                done = True
                reward = 1
                if hasattr(agent, "on_pre_training_done") and callable(agent.on_pre_training_done):
                    for elt in reached_goals:
                        if (elt == goal).all():
                            break
                    else:
                        reached_goals.append(goal)
            else:
                reward = -1

            # Ending time learning_step process ...
            _agent.on_action_stop(action, state, reward, done)

            # Store reward
            time_step_id += 1
        agent.episode_id += 1
        _agent.on_episode_stop()
        if isinstance(agent, STC_TL):
            agent.store_tc_training_samples()

    if hasattr(agent, "on_pre_training_done") and callable(agent.on_pre_training_done):
        start_state = environment.get_state(*environment.start_coordinates)
        agent.on_pre_training_done(start_state, reached_goals)
    print("Pre-training is done.")
