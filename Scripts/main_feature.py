import os
from feature_extraction import extract_features
from feature_selection import select_features
import numpy as np


data_dir = '/tmp/hmdb51/hmdb51_org/'
classes = os.listdir(data_dir)


with open("/tmp/Data/hmdb_features/class_label.txt", "w") as fl:
    for i in range(len(classes)):
        fl.write(classes[i]+str(i)+"\n")


video_label, video_name = extract_features(classes)
select_features(video_name, video_label)
