import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.batch_count = 1

    def get_batch(self, type_):
        if type_ == "test":
            data = np.load(self.config.test_path)
        elif type_ == "train": 
            data = np.load(self.config.train_path + self.batch_count)
            self.batch_count += 1
        elif type_ == "validation":
            data = np.load(self.config.train_path + 5)
        return data[:, 1:], data[:, 0]

    def next_batch(self, type_):
        if self.batch_count > 4:
            self.batch_count = 1
        X, Y = self.get_batch(type_)
        start, batch_size, tot = 0, self.config.batch_size, len(X)
        total = int(tot/ batch_size) # fix the last batch
        while start < total:
            end = start + batch_size
            x = X[start : end, :]
            y = Y[start : end, :]
            start += 1
            yield (x, y, int(total))