
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


def slice_bb(bb_file):
    with open(bb_path + bb_file + ".bb", "r") as f:
        start = []
        bb_data = []
        flag = 0
        for line in f:
            data = line.strip().split(" ")
            bb_data.append(data[1:5])
            if len(data) == 1:
                flag = 0
            if flag == 0 and len(data) != 1:
                start.append([int(data[0]), 0])
                flag = 1
                continue
            if flag == 1 and len(data[0]) != 1:
                start[-1][1] = int(data[0])
    bb_final = []
    for item in start:
        bb_final.append(bb_data[item[0]:item[1]+1])
    return bb_final, start


def extract_feature(samples, key, p):
    Z = []
    name_cnt = 0
    for cnt, name in enumerate(samples):
        name = name.split('\n')[0]
        print(name)
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
        bb_collec, span = slice_bb(bb_file)

        for counter, item in enumerate(span):
            bb_data = np.array(bb_collec[counter])
            index = np.arange(item[1] - item[0] + 1)

            index = GetSpacedElements(index)  # will return list now
            if index is None:
                continue
            print(bb_data.shape)
            print(span)
            bb_data = bb_data[index, :]
            index += item[0]
            M_new = M[index, :]
            X_new = X[index, :]
            for i in range(index.shape[0]):
                name_cnt += 1  # for naming purpose
                M_temp = M_new[i, :].reshape((10, 512, 7, 7))
                X_temp = X_new[i, :].reshape((10, 512, 7, 7))
                Z_temp = np.hstack((M_temp, X_temp))
                data = [Z_temp, key[cnt], bb_data[i]]
                name = str(key[cnt])+'_'+str(name_cnt)
                np_write(data, name, p)
                with open("./map_"+p+".txt", "a") as f:
                    f.write(bb_file + " " + name + " " + ",".join(str(e)
                                                                  for e in index[i])+"\n")


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
