import numpy as np
import h5py
from collections import Counter
from sklearn.model_selection import train_test_split

bb_path = "/tmp/virat_annotations/"


def GetSpacedElements(array, numElems=30):
    out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
    return np.array([out])

def np_write(data, name):
    path = 'Demo/Data/input/' + name + '.npy'
    np.save(path, data)


def extract_feature(samples, key):
    for cnt, name in enumerate(samples):
        name_cnt = 0
        M_path = 'Demo/Data/context_file/' + \
            str(name) + '.h5'
        X_path = 'Demo/Data/data_file/' + \
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
        bb_data = np.zeros(((end - start + 1), 4))

        with open(bb_path + vid_name, 'r') as f:
            for line in f:
                data = line.strip().split(" ")
                if key[cnt] == int(data[1]) and int(data[5]) >= start and int(data[5]) <= end:
                    bb_data[int(data[5]) - start] = list(map(int, data[6:]))

        for bb_cnt, temp in enumerate(bb_data):
            if np.array_equal(temp, np.array([0, 0, 0, 0])):
                bb_data[bb_cnt] = bb_data[bb_cnt-1]
            else:
                continue

        index = np.arange(end - start)
        if index.shape[0] > M.shape[0]:
            print('skip')
            continue
        index = GetSpacedElements(index)  # will return list now
        if index is None:
            continue

        bb_data = bb_data[1:, :]
        bb_data = bb_data[index, :]
        M_new = M[index, :]
        X_new = X[index, :]
        for i in range(index.shape[0]):
            name_cnt += 1  # for naming purpose
            M_temp = M_new[i, :].reshape((30, 512, 7, 7))
            X_temp = X_new[i, :].reshape((30, 512, 7, 7))
            Z_temp = np.hstack((M_temp, X_temp))
            data = [Z_temp, key[cnt], bb_data[i]]

            file_name = name + '_'+str(name_cnt)
            np_write(data, file_name)
            fp = open("Demo/map_demo_virat.txt", 'a')
            fp.write(file_name + " " + ",".join(str(q)
                                                for q in index[i]) + "\n")
