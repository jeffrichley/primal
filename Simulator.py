import numpy as np

class BasicSimulator():

    def __init__(self, state, width, height, **kwargs):

        self.state = state

        # screen information
        self.width = width
        self.height = height

        # sim information
        self.barrier_locations = state.barrier_locations
        self.asset_locations = state.asset_locations
        self.ally_locations = state.ally_locations
        self.enemy_locations = state.enemy_locations
        self.goal_locations = state.goal_locations

        # get the field ready to play
        self.state.reset_state()

    def move_asset(self, asset_number, new_location):
        # we need to make sure it is an empty location before we accept the move
        y = new_location[0]
        x = new_location[1]
        moved = True
        if self.state.is_empty_location(y, x):
            self.state.move_asset(asset_number, new_location)
        else:
            moved = False

        return moved

