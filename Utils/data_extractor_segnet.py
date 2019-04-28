import numpy as np
import os


class Data(object):
    def __init__(self):
        self.path = '/tmp/Data/segnet_input/'

    def get_split(self, dir):
        return os.listdir(self.path+dir)

    def get_data(self, n, dir):
        p = self.path + dir + '/' + n
        data = np.load(p)
        inp, att, label = data
        return inp, att, label
