import os
import numpy as np
from collections import Counter


class Dataset:

    def __init__(self, batch_size=32, sequence_length=30):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.batch_index = 0

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        # count and sort most frequent characters
        counter = Counter(data)
        self.sorted_chars = sorted(
            counter.keys(), key=counter.get, reverse=True)

        # self.sorted chars contains just the characters ordered descending by
        # frequency
        self.char2id = dict(
            zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return np.array([self.char2id[c] for c in sequence], dtype=np.int32)

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return [self.id2char[i] for i in encoded_sequence]

    def create_minibatches(self):
        data_len = len(self.x)
        chars_per_batch = self.batch_size * self.sequence_length
        self.num_batches = int((data_len-1) / chars_per_batch)

        self.batches = np.zeros(
            [self.num_batches, self.batch_size, self.sequence_length + 1],
            dtype=np.int32)
        for b in range(self.num_batches):
            for s in range(self.batch_size):
                sentance_start = s*(self.num_batches*self.sequence_length)
                start = b * self.sequence_length + sentance_start
                end = start + self.sequence_length + 1
                self.batches[b, s, :] = self.x[start:end]

        self.batch_index = 0

    def next_minibatch(self):
        new_epoch = self.batch_index == self.num_batches
        if new_epoch:
            self.batch_index = 0

        batch = self.batches[self.batch_index, :, :]
        self.batch_index += 1

        return new_epoch, batch[:, :-1], batch[:, 1:]
