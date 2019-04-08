import numpy as np
import h5py
from collections import Counter


def GetSpacedElements(array, numElems=30):
    number_seq = len(array)//numElems
    res = [[] for i in range(number_seq)]
    for i in range(number_seq*numElems):
        res[i%number_seq].append(i)
    return np.array(res)


def np_write(d, name):
    path = '/tmp/Data/virat_input/' + name + '.npy'
    np.save(path, d)


def extract_feature(samples, key):
    Z = []
    cnt = 0
    for name in samples:
        name = name.split('\n')[0]
        M_path = '/tmp/Data/virat_features/context_file/' + \
            str(name) + '.h5'
        X_path = '/tmp/Data/virat_features/data_file/' + \
            str(name) + '.h5'
        M_file = h5py.File(M_path, 'r')
        X_file = h5py.File(X_path, 'r')
        M = np.array(M_file.get('context_file'))
        X = np.array(X_file.get('data_file'))
        M_file.close()
        X_file.close()
        index = np.arange(len(M))
        index = GetSpacedElements(index) # will return list now
        M = M[index, :]
        X = X[index, :]
        for i in range(index.shape[0]):
            cnt += 1	# for naming purpose
            M_temp = M[i,:].reshape((30, 512, 7, 7))
            X_temp = X[i,:].reshape((30, 512, 7, 7))
            Z_temp = np.hstack((M_temp, X_temp))
            data = [Z_temp, key]
            name = str(key)+'_'+str(cnt)
            np_write(data, name)

def select_features(video_labels,video_names):
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
