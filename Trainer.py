import numpy as np
import tensorflow as tf

from RL_Stuff import ActorCriticModel
# from supervised import create_supervised_nn

class Trainer:

    def __init__(self, config, memory, simulator, state, model):
        self.memory = memory
        self.simulator = simulator
        self.state = state

        # training configuration
        self.ppo_epochs = config['ppo-epochs']
        self.max_grad_norm = config['max-grad-norm']
        self.training_size = config['training-size']

        self.asset_visible_window = config['asset-visible-window']

        rewards_config = config['rewards']
        self.basic_step = rewards_config['basic-step']
        self.ally_influence_step = rewards_config['ally-influence-step']
        self.enemy_influence_step = rewards_config['enemy-influence-step']
        self.mixed_influence_step = rewards_config['mixed-influence-step']
        self.goal_step = rewards_config['goal']

        network_config = config['network']
        self.state_shape = tuple(network_config['state-shape'])
        self.network_shape = tuple(network_config['network-shape'])
        self.goal_shape = tuple(network_config['goal-shape'])
        self.num_actions = network_config['number-of-actions']

        # self.supervised_model = create_supervised_nn(self.state_shape, self.goal_shape, self.num_actions)
        self.model = model

    def get_reward(self, asset_number, state):

        new_y, new_x = state.assets[asset_number]

        # we must check for mixed before enemy and ally influence because they would all be 1 if mixed
        reward = self.basic_step
        if state.goal_locations[new_y][new_x] == asset_number:
            reward = self.goal_step
        elif state.mixed_influence[new_y][new_x] == 1:
            reward = self.mixed_influence_step
        elif state.enemy_influence[new_y][new_x] == 1:
            reward = self.enemy_influence_step
        elif state.ally_influence[new_y][new_x] == 1:
            reward = self.ally_influence_step

        return reward

    def state_to_training(self, state, asset_number):

        # start out with everything zeros
        data = np.zeros(self.state_shape)

        # add influences
        data[0, :, :][(state.ally_influence == 1) & (state.mixed_influence != 1)] = 1
        data[1, :, :][(state.enemy_influence == 1) & (state.mixed_influence != 1)] = 1
        data[2, :, :][state.mixed_influence == 1] = 1

        # add walls
        data[3, :, :][state.barrier_locations == 1] = 1

        # add other assets
        current_asset = state.assets[asset_number]
        data[4, :, :][state.asset_locations == 1] = 1
        data[4, current_asset[0], current_asset[1]] = 0

        # add other's goals
        data[5, :, :][state.goal_locations >= 0] = 1
        data[5, :, :][state.goal_locations == asset_number] = 0

        # add our goals
        data[6, :, :][state.goal_locations == asset_number] = 1

        # add enemies and allies
        # TODO: Do we need to split the enemies and allies into separate frames?
        #       If not, the state space would be smaller.
        data[7, :, :][state.ally_locations == 1] = 1
        data[8, :, :][state.enemy_locations == 1] = 1

        # add where this asset is currently
        asset_location = state.get_asset_location(asset_number)
        data[9, asset_location[0], asset_location[1]] = 1

        # now we need to clip to only the viewing region
        # TODO: could probably make this faster by only making it the viewing region to begin with
        radius = self.asset_visible_window // 2

        # we pad everything with 0's except we need the barrier layer to be padded with 1's
        padded = np.pad(data, ((0, 0), (radius, radius), (radius, radius)), 'constant', constant_values=(0, 0))
        padded[3, :, :] = np.pad(data[3, :, :], ((radius, radius), (radius, radius)), 'constant', constant_values=(1, 1))

        # figure out where to slice
        offset_asset_location = (asset_location[0] + radius + 1, asset_location[1] + radius + 1)
        top_left_y = offset_asset_location[0] - radius - 1
        top_left_x = offset_asset_location[1] - radius - 1
        bottom_right_y = top_left_y + self.asset_visible_window
        bottom_right_x = top_left_x + self.asset_visible_window

        window_data = padded[:, top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # np.set_printoptions(linewidth=300)
        # print(data[3, :, :])
        # print(padded[3, :, :])

        # goal = np.array(state.goal_locations[1])
        # goal_location = np.where(state.goal_locations == asset_number)
        # goal_location = state.get_asset_goal_location(asset_number)
        goal = state.get_asset_goal_location(asset_number)

        # need to get a unit vector towards the goal
        dy = goal[0] - asset_location[0]
        dx = goal[1] - asset_location[1]
        magnitude = (dy**2 + dx**2)**0.5

        if magnitude != 0:
            goal_vector = (dy / magnitude, dx / magnitude)
        else:
            goal_vector = (dy, dx)

        return window_data, goal_vector

    def train(self):

        # TODO: for now we will just train with all data, perhaps we should just sample a certain number
        data = self.memory.sample_memory(self.training_size)

        # state, goal, action, reward
        cur_states = np.array([o[0] for o in data])
        cur_goals = np.array([o[1] for o in data])
        cur_actions = np.array([o[2] for o in data])
        rewards = np.array([o[3] for o in data])

        self.model.train_imitation(cur_states, cur_goals, cur_actions, rewards)

    def save_model(self, file_name):
        self.model.save_weights(file_name, overwrite=True)

    def load_model(self, file_name):
        self.model.load_weights(file_name)

    def predict_action(self, state, asset_number):

        state_data, goal_data = self.state_to_training(state, asset_number)
        # goal_data = state.goal_locations[asset_number]

        state_data = tf.convert_to_tensor([state_data])
        goal_data = tf.convert_to_tensor([goal_data])

        policy, value = self.model.call((state_data, goal_data))

        action = np.argmax(policy.numpy())

        # print(answers)

        return action
