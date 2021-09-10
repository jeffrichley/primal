import random
from tqdm import tqdm
import TrainingFactory
from DriverUtils import train_imitation, train_rl

imitation_probability = 0.30
# imitation_probability = 1.0

trainer, simulator, state, visualizer, memory, tester = TrainingFactory.create(config_file='./configs/training.yml')

# visualizer.visualize_combined_state(state)
# visualizer.visualize(state=state, show=True)

for sim_num in tqdm(range(1, 100001)):

    if sim_num < 100 or random.random() < imitation_probability:
        train_imitation(state, simulator, visualizer, trainer, memory)
    else:
        train_rl(state, simulator, visualizer, trainer, memory)



    state.reset_state()
    visualizer.reset()

    # the paper says they only train with the current round's data
    memory.reset()

    if sim_num % 100 == 0:
        trainer.save_model(f'./data/primal_weights_{sim_num}')

print('finished')

