import pickle
import os
import numpy as np


class Replay:
    def __init__(self, memory_size, batch_size, seed=0):
        self.rng = np.random.RandomState(seed)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        sampled_indices = [self.rng.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))

        return batch_data

    def sample_array(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [self.rng.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]

        return sampled_data

    def size(self):
        return len(self.data)

    def persist_memory(self, dir):
        for k in range(len(self.data)):
            transition = self.data[k]
            with open(os.path.join(dir, str(k)), "wb") as f:
                pickle.dump(transition, f)

    def clear(self):
        self.data = []
        self.pos = 0

    def get_buffer(self):
        return self.data
