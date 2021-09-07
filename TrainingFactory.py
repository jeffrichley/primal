from importlib import import_module
import yaml
from WorkingMemory import WorkingMemory
from PrimalState import PrimalState
from Trainer import Trainer
from Visualizer import Visualizer
from RL_Stuff import ActorCriticModel

def create(config_file):

    # Read the yml config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

        # create and configure the working memory
        memory_config = config['working-memory']
        memory_size = memory_config['size']
        memory = WorkingMemory(memory_size)

        # create and configure the state
        state = PrimalState(config)

        # create and configure the simulator
        simulator_config = config['simulator']
        simulator_type = simulator_config['type']
        module_path, class_name = simulator_type.rsplit('.', 1)
        module = import_module(module_path)
        simulator = getattr(module, class_name)(state, **simulator_config)

        # create a model
        trainer_config = config['trainer']
        model = ActorCriticModel(trainer_config)

        # create and configure the trainer

        trainer = Trainer(trainer_config, memory, simulator, state, model)

        # create and configure the visualizer
        visualizer = Visualizer(config['visualization'])

    return trainer, simulator, state, visualizer, memory
