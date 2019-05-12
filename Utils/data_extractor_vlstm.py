import numpy as np
import os


class Data(object):
    def __init__(self):
        self.path = '/tmp/Data/virat_input/'

    def get_split(self, dir):
        return os.listdir(self.path+dir)

    def get_data(self, n, dir):
        p = self.path + dir + '/' + n
        Z = np.load(p)
        Z, y, _ = Z
        Z = Z.reshape((30, 7, 7, 1024))
        return Z, y
