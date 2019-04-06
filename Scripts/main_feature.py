import os
from feature_extraction import extract_features
from feature_selection import select_features



data_dir = '/home/user/Downloads/Crime-Dataset/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Videos/'
classes = os.listdir(data_dir)



with open("Data/crime_data_features/class_label.txt", "w") as fl:
    for i in range(len(classes)):
        fl.write(classes[i]+str(i)+"\n")



for index,item in enumerate(classes):
    video_label,video_name = extract_features([item],index)
    select_features(video_label,video_name)
    os.system(" rm -rf ./Data/crime_data_features/data_file")
    os.system(" rm -rf ./Data/crime_data_features/context_file")
    os.system("mkdir ./Data/crime_data_features/data_file")
    os.system("mkdir ./Data/crime_data_features/context_file")
     
    











