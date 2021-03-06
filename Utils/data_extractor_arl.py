import numpy as np
import os


class Data(object):
    def __init__(self):
        self.path = '/tmp/Data/hmdb_input/'

    def get_split(self, dir):
        '''
        file_path = 'Data/virat_'+dir+'.txt'
        tbr = []
        with open(file_path, 'r') as myfile:
            for line in myfile:
                tbr.append(line.strip())
        return tbr
        '''
        return os.listdir(self.path+dir)

    def get_data(self, n, dir):
        p = self.path + dir + '/' + n
        Z = np.load(p)
        Z, y, _ = Z
        Z = Z.reshape((10, 7, 7, 1024))
        return Z, y
