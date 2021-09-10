import numpy as np

from PathingExpert import a_star


def train_rl(state, simulator, visualizer, trainer, memory):

    # TODO: ew, this is the fourth place this showed up, need to fix
    FOUR_CONNECTED_MOVES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

    num_assets = len(state.assets)
    num_actions = len(FOUR_CONNECTED_MOVES)

    next_values = [0 for _ in range(num_assets)]
    cur_rewards = [[] for _ in range(num_assets)]
    cur_values = [[] for _ in range(num_assets)]
    cur_states = [[] for _ in range(num_assets)]
    cur_valids = [[] for _ in range(num_assets)]
    cur_goals = [[] for _ in range(num_assets)]

    going = True
    sim_round = 0
    while going:
        sim_round += 1
        # lets make sure we don't go on forever if we get stuck
        if sim_round > 200:
            # visualizer.visualize_combined_state(state)
            break

        going = False

        for dude_num in range(len(state.assets)):
            asset = state.assets[dude_num]
            goal = state.goals[dude_num]

            # TODO: may need to add when they are on the goal too to learn stick actions
            # if asset != goal:
            if True:
                # cost = state.get_cost_matrix()

                # if agent_paths[dude_num] is None and agent_actions[dude_num] is None:
                #     path, actions = a_star(asset, goal, cost)
                #     agent_paths[dude_num] = path
                #     agent_actions[dude_num] = actions
                #     # get rid of current location
                #     path.pop(0)
                # else:
                #     path = agent_paths[dude_num]
                #     actions = agent_actions[dude_num]

                if asset != goal:
                    going = True

                # new_location = path.pop(0)
                # moved = simulator.move_asset(dude_num, new_location)

                valid_moves = [1 for i in range(num_actions)]
                for i in range(num_actions):
                    test_move = FOUR_CONNECTED_MOVES[i]
                    test_location = (asset[0] + test_move[0], asset[1] + test_move[1])
                    valid_moves[i] = int(simulator.valid_action(dude_num, test_location))

                predicted_action, value = trainer.predict_action(state, dude_num)
                action = trainer.epsilon_greedy_action(predicted_action)
                move = FOUR_CONNECTED_MOVES[action]
                new_location = (asset[0] + move[0], asset[1] + move[1])
                moved = simulator.move_asset(dude_num, new_location)

                if moved:
                    visualizer.move_asset(asset, new_location)
                # else:
                #     path, actions = a_star(asset, goal, cost)
                #     agent_paths[dude_num] = path
                #     agent_actions[dude_num] = actions
                #     # get rid of current location
                #     path.pop(0)
                #
                #     # try going again to the next best place
                #     new_location = path.pop(0)
                #     moved = simulator.move_asset(dude_num, new_location)

                state_data, goal_data = trainer.state_to_training(state, dude_num)
                # if len(actions) == 0:
                #     action = 0
                # else:
                #     action = actions.pop(0)



                reward = trainer.get_reward(dude_num, state)

                cur_values[dude_num].append(value)
                cur_rewards[dude_num].append(reward)
                cur_states[dude_num].append(state_data)
                # cur_valids[dude_num].append(int(moved))
                cur_valids[dude_num].append(valid_moves)
                cur_goals[dude_num].append(goal_data)

    for dude_num in range(len(state.assets)):
        asset = state.assets[dude_num]
        goal = state.goals[dude_num]
        action, next_value = trainer.predict_action(state, dude_num)
        move = FOUR_CONNECTED_MOVES[action]
        new_location = (asset[0] + move[0], asset[1] + move[1])
        moved = simulator.move_asset(dude_num, new_location)

        next_values[dude_num] = next_value

    # returns = np.array([compute_gae(next_values[asset], cur_rewards[asset], masks, cur_values[asset]) for asset in range(NUM_ASSETS)])
    returns = np.array([trainer.compute_gae(next_values[asset], cur_rewards[asset], 1, cur_values[asset]) for asset in range(num_assets)])

    advantages = trainer.normalize_returns(returns - cur_values)

    # state, goal, action, reward
    # memory.remember(state_data, goal_data, action, reward)

    flatten = lambda t: [item for sublist in t for item in sublist]

    cur_states = flatten(cur_states)
    returns = flatten(returns)
    advantages = flatten(advantages)
    cur_valids = flatten(cur_valids)
    cur_goals = flatten(cur_goals)

    # TODO: we may be training too many times on the same information,
    #       we might need to look into training every so many sessions
    trainer.train_rl(cur_states, returns, advantages, cur_valids, cur_goals)


def train_imitation(state, simulator, visualizer, trainer, memory):

    agent_actions = [None for i in range(len(state.assets))]
    agent_paths = [None for i in range(len(state.assets))]

    going = True
    sim_round = 0
    while going:
        sim_round += 1
        # lets make sure we don't go on forever if we get stuck
        if sim_round > 200:
            # visualizer.visualize_combined_state(state)
            break

        going = False

        for dude_num in range(len(state.assets)):
            asset = state.assets[dude_num]
            goal = state.goals[dude_num]

            # TODO: may need to also add for when they are on the goal for stick actions
            if asset != goal:
                cost = state.get_cost_matrix()

                if agent_paths[dude_num] is None and agent_actions[dude_num] is None:
                    path, actions = a_star(asset, goal, cost)
                    agent_paths[dude_num] = path
                    agent_actions[dude_num] = actions
                    # get rid of current location
                    path.pop(0)
                else:
                    path = agent_paths[dude_num]
                    actions = agent_actions[dude_num]

                if len(path) > 0:

                    going = True

                    new_location = path.pop(0)
                    moved = simulator.move_asset(dude_num, new_location)

                    if moved:
                        visualizer.move_asset(asset, new_location)
                    else:
                        path, actions = a_star(asset, goal, cost)
                        agent_paths[dude_num] = path
                        agent_actions[dude_num] = actions
                        # get rid of current location
                        path.pop(0)

                        # try going again to the next best place
                        new_location = path.pop(0)
                        moved = simulator.move_asset(dude_num, new_location)

                    state_data, goal_data = trainer.state_to_training(state, dude_num)
                    if len(actions) == 0:
                        action = 0
                    else:
                        action = actions.pop(0)
                    reward = trainer.get_reward(dude_num, state)

                    # state, goal, action, reward
                    memory.remember(state_data, goal_data, action, reward)

    # TODO: we may be training too many times on the same information,
    #       we might need to look into training every so many sessions
    trainer.train()
