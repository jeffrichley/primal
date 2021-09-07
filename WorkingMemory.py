from random import random, sample

# Uses a basic array for holding previous training information.
# This will hold a certain max amount of training episodes and then start forgetting the oldest
class WorkingMemory:

    def __init__(self, memory_size=100000):
        # how many training samples should we keep?
        self.memory_size = memory_size

        # our actual memory to hold the training samples
        self.memory = []

    # add a training sample to our memory
    def remember(self, state, goal, action, reward):
        # add the training sample
        # training_example = (s, a, r, s_prime, done)
        training_example = (state, goal, action, reward)
        self.memory.append(training_example)

        # if we are over the limit, start forgetting
        if len(self.memory) > self.memory_size:
            # pop from the beginning because that is the oldest
            self.memory.pop(0)

    # get a bunch of samples to train on
    def sample_memory(self, count):
        # return the number they asked for or the entire memory, which ever is smaller
        return sample(self.memory, min(count, len(self.memory)))

    def num_samples(self):
        return len(self.memory)