import numpy as np

class BasicSimulator():

    def __init__(self, state, width, height, **kwargs):

        self.state = state

        # screen information
        # self.width = width
        # self.height = height

        # sim information
        # self.barrier_locations = state.barrier_locations
        # self.asset_locations = state.asset_locations
        # self.ally_locations = state.ally_locations
        # self.enemy_locations = state.enemy_locations
        # self.goal_locations = state.goal_locations

        # get the field ready to play
        self.state.reset_state()

    def move_asset(self, asset_number, new_location):
        # we need to make sure it is an empty location before we accept the move
        moved = True
        if self.valid_action(new_location):
            self.state.move_asset(asset_number, new_location)
        else:
            moved = False

        return moved

    def valid_action(self, new_location):

        y = new_location[0]
        x = new_location[1]
        return self.state.is_empty_location(y, x)


