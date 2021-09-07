import TrainingFactory
import time
from tqdm import tqdm

from PathingExpert import a_star


trainer, simulator, state, visualizer, memory = TrainingFactory.create(config_file='./configs/training.yml')
# trainer.train()

# visualizer.visualize_combined_state(state)

# visualizer.visualize(state=state, show=True)
# asset = state.assets[0]
# goal = state.goals[0]

# time.sleep(5)

# for i in range(20):
#     start = time.time()
#     state.reset_state()
#     end = time.time()
#     print(f'{i} took {end - start}')

for sim_num in tqdm(range(10001)):

    start = time.time()

    agent_actions = [None for i in range(len(state.assets))]
    agent_paths = [None for i in range(len(state.assets))]

    going = True
    round = 0
    while going:
        round += 1
        # lets make sure we don't go on forever if we get stuck
        if round > 200:
            # visualizer.visualize_combined_state(state)
            # print('too many rounds')
            break

        going = False

        # if round == 199:
        #     visualizer.visualize_combined_state(state)

        for dude_num in range(len(state.assets)):
            asset = state.assets[dude_num]
            goal = state.goals[dude_num]

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
                    # new_location = path[1]
                    # new_location = path.pop(0)

                    # print(dude_num, new_location, path)

                    new_location = path.pop(0)
                    moved = simulator.move_asset(dude_num, new_location)


                    if moved:
                        visualizer.move_asset(asset, new_location)
                        # time.sleep(0.1)
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
                    # action = actions[0]
                    if len(actions) == 0:
                        action = 0
                    else:
                        action = actions.pop(0)
                    reward = trainer.get_reward(dude_num, state)

                    # state, goal, action, reward
                    memory.remember(state_data, goal_data, action, reward)

                    # if dude_num == 0:
                    #     reward = trainer.get_reward(dude_num, state)
                    #     print(reward, state.assets[dude_num], goal)

    end = time.time()
    # print(f'\nsim {sim_num} took {end - start}')

    # start = time.time()

    # let the pump get primed so we don't memorize the first handful and look like we are
    # doing much better than we really are
    # TODO: we may be training too many times on the same information, we might need to look into training every so many sessions
    if memory.num_samples() > 10000:
        trainer.train()
    # end = time.time()
    # print(f'\ntraining {sim_num} took {end - start} with memory size of {len(memory.memory)}')

    state.reset_state()
    visualizer.reset()

    if sim_num % 100 == 0:
        trainer.save_model(f'./data/primal_weights_{sim_num}')

    # time.sleep(0.1)

print('finished')

