import numpy as np
import h5py
from collections import Counter
from sklearn.model_selection import train_test_split

bb_path = "/tmp/hmdb51/bb_file/HMDB51/"


def GetSpacedElements(array, numElems=10):
    if len(array) < numElems:
        return None
    number_seq = len(array)//numElems
    res = [[] for i in range(number_seq)]
    for i in range(number_seq*numElems):
        res[i % number_seq].append(i)
    return np.array(res)


def np_write(data, name, p):
    path = '/tmp/Data/hmdb_input/' + p + '/' + name + '.npy'
    np.save(path, data)


def extract_feature(samples, key, p):
    Z = []
    name_cnt = 0
    for cnt, name in enumerate(samples):
        name = name.split('\n')[0]
        M_path = '/tmp/Data/hmdb_features/context_file/' + \
            str(name) + '.h5'
        X_path = '/tmp/Data/hmdb_features/data_file/' + \
            str(name) + '.h5'
        M_file = h5py.File(M_path, 'r')
        X_file = h5py.File(X_path, 'r')
        M = np.array(M_file.get('context_file'))
        X = np.array(X_file.get('data_file'))
        M_file.close()
        X_file.close()

        bb_file = name.split('.')[0]
        with open(bb_path + bb_file+".bb", "r") as f:
            start = 0
            end = 0
            flag = 0
            bb_data = []
            for line in f:
                data = line.strip().split(" ")
                if flag == 0 and len(data) != 1:
                    start = int(data[0])
                    flag = 1
                    bb_data.append(data[1:])
                    continue
                if flag == 1 and len(data) != 1:
                    end = int(data[0])
                    bb_data.append(data[1:])

        bb_data = np.array(bb_data)
        print(bb_data.shape)
        index = np.arange(end - start + 1)
        print(index)
        index = GetSpacedElements(index)  # will return list now
        if index is None:
            continue
        print(index)
        bb_data = bb_data[index, :]
        index += start
        M = M[index, :]
        X = X[index, :]

        for i in range(index.shape[0]):
            name_cnt += 1  # for naming purpose
            M_temp = M[i, :].reshape((10, 512, 7, 7))
            X_temp = X[i, :].reshape((10, 512, 7, 7))
            Z_temp = np.hstack((M_temp, X_temp))
            data = [Z_temp, key[cnt], bb_data[i]]
            name = str(key[cnt])+'_'+str(name_cnt)
            np_write(data, name, p)


def select_features(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42)

    extract_feature(X_train, y_train, 'train')
    extract_feature(X_val, y_val, 'val')
    extract_feature(X_test, y_test, 'test')

    '''
    vid_dict = {}
    for i in range(len(video_labels)):
        key = video_labels[i]
        value = video_names[i]
        if key in vid_dict:
            temp = vid_dict.get(key)
            temp.append(value)
            vid_dict[key] = temp
        else:
            vid_dict[key] = [value]

    for key in vid_dict.keys():
        vid_list = vid_dict.get(key)
        np.random.shuffle(vid_list)
        samples = vid_list
        extract_feature(samples, key)
    '''
