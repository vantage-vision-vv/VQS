import numpy as np
import h5py
from collections import Counter
from sklearn.model_selection import train_test_split

bb_path = "/tmp/virat_annotations/"


def GetSpacedElements(array, numElems=10):
    if len(array) < numElems:
        return None
    number_seq = len(array)//numElems
    res = [[] for i in range(number_seq)]
    for i in range(number_seq*numElems):
        res[i % number_seq].append(i)
    return np.array(res)


def np_write(data, name, p):
    path = '/tmp/Data/virat_input/' + p + '/' + name + '.npy'
    np.save(path, data)


def extract_feature(samples, key, p):
    for cnt, name in enumerate(samples):
        name_cnt = 0
        M_path = '/tmp/Data/virat_input/context_file/' + \
            str(name) + '.h5'
        X_path = '/tmp/Data/virat_input/data_file/' + \
            str(name) + '.h5'
        M_file = h5py.File(M_path, 'r')
        X_file = h5py.File(X_path, 'r')
        M = np.array(M_file.get('context_file'))
        X = np.array(X_file.get('data_file'))
        M_file.close()
        X_file.close()

        start = int(name.split("_")[-3])
        end = int(name.split("_")[-2])

        vid_name = "_".join(name.split("_")[1:-4])
        vid_name += ".viratdata.events.txt"
        bb_data = np.zeros(((end - start + 1),4))
        with open(bb_path + vid_name, 'r') as f:
            for line in f:
                data = line.strip().split(" ")
                if key[cnt] == int(data[1]) and int(data[5]) >= start and int(data[5]) <= end:
                    bb_data[int(data[5]) - start] = list(map(int,data[6:]))
        for bb_cnt, temp in enumerate(bb_data):
            if np.array_equal(temp, np.array([0,0,0,0])):
                bb_data[bb_cnt] = bb_data[bb_cnt-1]
            else:
                continue
        index = np.arange(end - start)
        index = GetSpacedElements(index)  # will return list now
        if index is None:
            continue
        bb_data = bb_data[1:,:]
        bb_data = bb_data[index, :]
        M_new = M[index, :]
        X_new = X[index, :]
        for i in range(index.shape[0]):
            name_cnt += 1  # for naming purpose
            M_temp = M_new[i, :].reshape((10, 512, 7, 7))
            X_temp = X_new[i, :].reshape((10, 512, 7, 7))
            Z_temp = np.hstack((M_temp, X_temp))
            data = [Z_temp, key[cnt], bb_data[i]]
            file_name = name + '_'+str(name_cnt)
            np_write(data, file_name, p)


def select_features(X, y):
    data = {}
    for item in X:
        if item.split("_")[0] in data.keys():
            temp = data.get(item.split("_")[0])
            temp.append(item)
            data[item.split("_")[0]] = temp
        else:
            data[item.split("_")[0]] = [item]
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []
    for i in data.keys():
        y = np.full((len(data.get(i)),), int(i))
        a, b, c, d = train_test_split(
            data[i], y, test_size=0.2, random_state=42)
        X_test.extend(b)
        y_test.extend(d)
        x, y, z, w = train_test_split(a, c, test_size=0.25, random_state=42)
        X_train.extend(x)
        X_val.extend(y)
        y_train.extend(z)
        y_val.extend(w)
    print(len(X_train),len(y_train))
    extract_feature(X_train, y_train, 'train')
    extract_feature(X_val, y_val, 'val')
    extract_feature(X_test, y_test, 'test')
