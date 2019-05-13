import os
from demo_feature_extraction import extract_features
from demo_feature_selection import extract_feature
import numpy as np


vid_dir = 'Demo/trim_videos/'
vid_files = os.listdir(vid_dir)
video_label, video_name = extract_features(vid_files)
extract_feature(video_name, video_label)
