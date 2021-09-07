import pickle
import copy

class PrimalTester:

    def __init__(self, trainer, simulator, state, model):
        # trainer, simulator, state, visualizer, memory, tester = TrainingFactory.create(config_file=config_file)

        self.trainer = copy.copy(trainer)
        self.simulator = copy.copy(simulator)
        # self.model = model
        # self.original_state = state

        # inject the real model
        # self.trainer.model = model

    def score(self, num_sims=5):

        # TODO: ew, this is the third place this showed up, need to fix
        FOUR_CONNECTED_MOVES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

        score = 0

        for sim_num in range(1, num_sims + 1):
            with open(f'./test_data/test_sim_{sim_num}.pkl', 'rb') as inp:
                sim_state = pickle.load(inp)

            # inject our test scenario into the mix
            self.trainer.state = sim_state
            self.simulator.state = sim_state

            going = True
            sim_round = 0

            while going and sim_round < 200:

                sim_round += 1

                going = False

                for dude_num in range(len(sim_state.assets)):
                    asset = sim_state.assets[dude_num]
                    goal = sim_state.goals[dude_num]

                    if asset == goal:
                        pass
                    else:
                        going = True

                        action_number = self.trainer.predict_action(sim_state, dude_num)

                        move = FOUR_CONNECTED_MOVES[action_number]

                        new_location = (asset[0] + move[0], asset[1] + move[1])

                        self.simulator.move_asset(dude_num, new_location)

                        reward = self.trainer.get_reward(dude_num, sim_state)
                        score += reward

        # put things back to normal, just like we found it
        # self.trainer.state = self.original_state
        # self.simulator.state = self.original_state

        return score


if __name__ == '__main__':
    import TrainingFactory

    trainer, simulator, state, visualizer, memory, tester = TrainingFactory.create(config_file='./configs/training.yml')

    trainer.load_model('./models/exp1')

    print('model score:', tester.score())