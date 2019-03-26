import numpy as np
import h5py
from collections import Counter


def GetSpacedElements(array, numElems=30):
    out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
    return out


def h5_write(d, name):
    path = '/home/alpha/Work/VQS/Data/hmdb51_input/' + name + '.npy'
    #hf = h5py.File(path, 'w')
    #hf.create_dataset(name, data=d)
    # hf.close()
    np.save(path, d)


def extract_feature(samples, key):
    Z = []
    cnt = 0
    for name in samples:
        cnt += 1
        name = name.split('\n')[0]
        M_path = '/home/alpha/Work/VQS/Data/hmdb51_features/context_file/' + \
            str(name) + '.h5'
        X_path = '/home/alpha/Work/VQS/Data/hmdb51_features/data_file/' + \
            str(name) + '.h5'
        M_file = h5py.File(M_path, 'r')
        X_file = h5py.File(X_path, 'r')
        M = np.array(M_file.get('context_file'))
        X = np.array(X_file.get('data_file'))
        M_file.close()
        X_file.close()
        index = np.arange(len(M))
        index = GetSpacedElements(index)
        M = M[index, :]
        X = X[index, :]
        M = M.reshape((30, 512, 7, 7))
        X = X.reshape((30, 512, 7, 7))
        Z_temp = np.hstack((M, X))
        data = [Z_temp, key]
        name = str(key)+'_'+str(cnt)
        h5_write(data, name)


if __name__ == '__main__':
    video_label_path = '/home/alpha/Work/VQS/Data/hmdb51_features/video_label.txt'
    video_name_path = '/home/alpha/Work/VQS/Data/hmdb51_features/video_name.txt'
    frame_number_path = '/home/alpha/Work/VQS/Data/hmdb51_features/frame_number.txt'

    with open(video_label_path, 'r') as myfile:
        video_labels = myfile.readlines()
        video_labels = np.array(video_labels).astype(int)

    with open(video_name_path, 'r') as myfile:
        video_names = myfile.readlines()
        video_names = np.array(video_names).astype(str)

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

    class_num = Counter(video_labels)
    min_key = min(class_num, key=class_num.get)
    clips_bottleneck = class_num.get(min_key)

    Z = []
    y = []
    for key in vid_dict.keys():
        vid_list = vid_dict.get(key)
        np.random.shuffle(vid_list)
        samples = vid_list[:clips_bottleneck]
        extract_feature(samples, key)
