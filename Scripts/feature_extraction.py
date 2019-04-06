import cv2
import os
import numpy as np
import h5py
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


model = VGG16(weights='imagenet', include_top=False)

data_dir = '/home/user/Downloads/Crime-Dataset/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Videos/'

def compute_rgb(image):
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature.flatten()


def compute_flow(curr, prev):
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    channels = []
    for i in range(2):
        x = flow[:, :, i]
        x[x > 20] = 20
        x[x < -20] = -20
        domain = np.min(x), np.max(x)
        x = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        channels.append(x)
    temp = np.zeros_like(channels[0])
    stack = np.stack(
        [np.array(channels[0]), np.array(channels[1]), temp], axis=-1)
    return compute_rgb(stack.reshape((224, 224, 3)))


def extract_features(classes,label):
    video_label = []
    video_name = []
    for item in classes:
        print(item)
        files = os.listdir(data_dir+item+"/")
        cnt = 0
        for vid in files:
            cnt += 1
            rgb_features = []
            flow_features = []
            cap = cv2.VideoCapture(data_dir+item+"/"+vid)
            frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            chk, initial_frame = cap.read()
            if chk is False:
                continue
            initial_frame = cv2.resize(
                initial_frame, (224, 224), interpolation=cv2.INTER_AREA)
            for i in range(frame_length-1):
                chk, frame = cap.read()
                if chk is False:
                    continue
                image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                rgb_features.append(compute_rgb(image))
                flow_features.append(compute_flow(image, initial_frame))
                initial_frame = image
            video_label.append(label)
            video_name.append(vid)
            hf_rgb = h5py.File("Data/crime_data_features/data_file/"+vid+".h5", 'w')
            hf_flow = h5py.File("Data/crime_data_features/context_file/"+vid+".h5", 'w')
            hf_rgb.create_dataset('data_file', data=rgb_features)
            hf_flow.create_dataset('context_file', data=flow_features)
            hf_rgb.close()
            hf_flow.close()
            if cnt % 10 == 0:
                print(cnt)
            else:
                print('.', end='')
        print(item + " class completed")
    return video_label,video_name


