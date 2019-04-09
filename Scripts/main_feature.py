import os
from feature_extraction import extract_features
from feature_selection import select_features



data_dir = '/tmp/Virat/'
classes = os.listdir(data_dir)
classes = classes.sort()


with open("/tmp/Data/virat_features/class_label.txt", "w") as fl:
    for i in range(len(classes)):
        fl.write(classes[i]+str(i)+"\n")



for index,item in enumerate(classes):
    if item != 'Carry':
        video_label,video_name = extract_features([item],index)
    select_features(video_label,video_name)
    os.system(" rm -rf /tmp/Data/virat_features/data_file")
    os.system(" rm -rf /tmp/Data/virat_features/context_file")
    os.system("mkdir /tmp/Data/virat_features/data_file")
    os.system("mkdir /tmp/Data/virat_features/context_file")
     
    











