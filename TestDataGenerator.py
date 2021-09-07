import TrainingFactory
import pickle

trainer, simulator, state, visualizer, memory, _ = TrainingFactory.create(config_file='./configs/training.yml')


for sim_num in range(1, 11):

    visualizer.visualize_combined_state(state, file_name=f'./test_data/test_sim_img_{sim_num}.jpg')

    with open(f'./test_data/test_sim_{sim_num}.pkl', 'wb') as outp:
        pickle.dump(state, outp, pickle.HIGHEST_PROTOCOL)

    state.reset_state()

