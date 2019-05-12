import os
import numpy as np
import random
import csv

dir = '/tmp/Data/virat_input/'


def gen_list(p):
    path = dir + p + '/'
    f = os.listdir(path)
    data = {}
    for fi in f:
        key = int(fi.split('_')[0])
        if key in data.keys():
            temp = data.get(key)
            temp.append(fi)
            data[key] = temp
        else:
            data[key] = [fi]

    count = []
    for key in data.keys():
        temp = data.get(key)
        count.append(len(temp))
    s = np.sum(count) - np.max(count)
    a = s // (len(count) - 1)

    tbr = []
    for key in data.keys():
        temp = data.get(key)
        if len(temp) <= a:
            tbr.extend(temp)
        else:
            random.shuffle(temp)
            tbr.extend(temp[0:a])
    return tbr


def write(p):
    with open('Data/virat_'+p, 'w') as myfile:
        l = gen_list(p)
        writer = csv.writer(myfile)
        writer.writerows(l)
    return None

if __name__ == "__main__":
    write('train')
    write('test')
    write('val')