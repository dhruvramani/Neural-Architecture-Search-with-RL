import os
import utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.batch_count = 1

    def load_data(self, file_name):
        with open(file_name, 'rb') as file:
            unpickler = pickle._Unpickler(file)
            unpickler.encoding = 'latin1'
            contents = unpickler.load()
            X, Y = np.asarray(contents['data'], dtype=np.float32), np.asarray(contents['labels'])
            one_hot = np.zeros((Y.size, Y.max() + 1))
            one_hot[np.arange(Y.size), Y] = 1
            return X, one_hot

    def get_batch(self, type_):
        if type_ == "test":
            return self.load_data(self.config.test_path)
        elif type_ == "train": 
            self.batch_count += 1
            return self.load_data(self.config.train_path + str(self.batch_count))
        elif type_ == "validation":
            return self.load_data(self.config.train_path + "5")

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