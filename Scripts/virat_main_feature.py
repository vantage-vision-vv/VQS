import os
from virat_feature_extraction import extract_features
from virat_feature_selection import select_features
import numpy as np


vid_dir = '/home/alpha/Work/Dataset/Virat_Ground/Virat_Trimed/'
vid_files = os.listdir(vid_dir)
video_label, video_name = extract_features(vid_files)
select_features(video_name, video_label)
