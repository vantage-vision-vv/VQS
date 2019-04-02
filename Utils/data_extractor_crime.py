import numpy as np
import os
from sklearn.model_selection import train_test_split


class Data(object):
    def __init__(self):
        self.path = 'Data/crime_input'
        self.dir = os.listdir(self.path)

    def get_split(self):
        train, test = train_test_split(
            self.dir, test_size=0.2, random_state=42)
        train, val = train_test_split(train, test_size=0.2, random_state=42)
        return train, val, test

    def get_data(self, n):
        p = self.path + '/' + n
        Z = np.load(p)
        Z, y = Z
        Z = Z.reshape((30,7,7,1024))
        return Z, y
