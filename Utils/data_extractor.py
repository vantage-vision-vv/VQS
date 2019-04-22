import numpy as np
import os
from sklearn.model_selection import train_test_split


class Data(object):
    def __init__(self):
        self.path = '/tmp/Data/hmdb_input/'

    def get_split(self,dir):
        return os.listdir(self.path+dir)

    def get_data(self, n, dir):
        p = self.path + dir + '/' + n
        Z = np.load(p)
        Z, y, _ = Z
        Z = Z.reshape((10,7,7,1024))
        return Z, y
