import os
from feature_extraction import extract_features
from feature_selection import select_features
import numpy as np


d = 'train'
data_dir = '/tmp/Hollywood/' + d + '/'
classes = os.listdir(data_dir)


with open("/tmp/Data/hollywood_features/" + d + "/class_label.txt", "w") as fl:
    for i in range(len(classes)):
        fl.write(classes[i]+str(i)+"\n")


video_label, video_name = extract_features(classes)
select_features(video_label, video_name)
