import os
from feature_extraction import extract_features
from feature_selection import select_features
import numpy as np
from pathlib import Path

d = 'train'
data_dir = '/tmp/Hollywood/' + d + '/'
classes = os.listdir(data_dir)


with open("/tmp/Data/hollywood_features/" + d + "/class_label.txt", "w") as fl:
    for i in range(len(classes)):
        fl.write(classes[i]+str(i)+"\n")

'''
for index, item in enumerate(classes):
    video_label, video_name = extract_features([item], index)
'''
video_name = np.array(list(Path(
    "/tmp/Data/hollywood_features/" + d + '/').rglob("*.[a][v][i].*"))).astype(str)
video_label = []
for i in range(video_name.shape[0]):
    video_name[i] = video_name[i].split('/')[-1]
    video_label.append(int(video_name[i].split('_')[0]))

select_features(video_label, video_name)
