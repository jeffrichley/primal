import TrainingFactory
import time
from tqdm import tqdm

from PathingExpert import a_star


trainer, simulator, state, visualizer, memory, tester = TrainingFactory.create(config_file='./configs/training.yml')

trainer.load_model('./data/primal_weights_6200')

visualizer.visualize(state=state, show=True)

# TODO: ew, this is the second place this showed up, need to fix
FOUR_CONNECTED_MOVES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

print('starting...')

time.sleep(5)

going = True
round = 0
while going:

    num_on_goal = 0

    round += 1
    # lets make sure we don't go on forever if we get stuck
    if round > 400:
        break;

    going = False

    for dude_num in range(len(state.assets)):
        asset = state.assets[dude_num]
        goal = state.goals[dude_num]

        if asset == goal:
            # print(f'bang-a-rang asset {dude_num} made it!')
            num_on_goal += 1
        else:
            going = True

            action_number, value = trainer.predict_action(state, dude_num)

            move = FOUR_CONNECTED_MOVES[action_number]

            new_location = (asset[0] + move[0], asset[1] + move[1])

            moved = simulator.move_asset(dude_num, new_location)
            if moved:
                visualizer.move_asset(asset, new_location)

            time.sleep(0.1)

            # if dude_num == 0:
            #     print(f'{dude_num} going to {goal} moved {asset} -> {new_location}')


    print(num_on_goal)


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

            # if len(path) > 0:

                # going = True
                #
                #
                # new_location = path.pop(0)
                # moved = simulator.move_asset(dude_num, new_location)
                #
                #
                # if moved:
                #     visualizer.move_asset(asset, new_location)
                #
                # else:
                    # path, actions = a_star(asset, goal, cost)
                    # agent_paths[dude_num] = path
                    # agent_actions[dude_num] = actions

                    # get rid of current location
                    # path.pop(0)

                    # try going again to the next best place
                    # new_location = path.pop(0)
                    # moved = simulator.move_asset(dude_num, new_location)





                # state_data, goal_data = trainer.state_to_training(state, dude_num)
                # # action = actions[0]
                # if len(actions) == 0:
                #     action = 0
                # else:
                #     action = actions.pop(0)
                # reward = trainer.get_reward(dude_num, state)


                # memory.remember(state_data, goal_data, action, reward)





end = time.time()
# print(f'\nsim {sim_num} took {end - start}')

# start = time.time()
trainer.train()
# end = time.time()
# print(f'\ntraining {sim_num} took {end - start} with memory size of {len(memory.memory)}')

state.reset_state()
visualizer.reset()



print('finished')

