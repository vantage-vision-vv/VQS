import numpy as np
import os


class Data(object):
    def __init__(self):
        self.path = '/tmp/Data/hmdb_input/'

    def get_split(self, dir):
        dir = dir + '_attn'
        return os.listdir(self.path+dir)

    def get_data(self, n, dir):
        dir = dir + '_attn'
        p = self.path + dir + '/' + n
        data = np.load(p)
        inp, att, label = data
        return inp, att, label
