import cv2
import os
import numpy as np
import h5py
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


model = VGG16(weights='imagenet', include_top=False)

data_dir = "/tmp/Virat_Trimed/"

annotation_path = "/tmp/virat_annotations/"

max_width = 683
max_height = 343


def crop_frame(vid_name, frame_no, frame, width, height, prev_bb):
    vid_name = vid_name.split("_")
    ann_file = "_".join(vid_name[1:-4]) + ".viratdata.events.txt"
    bb_data = []
    with open(annotation_path + ann_file, "r") as f:
        for line in f:
            data = line.strip().split(" ")
            if int(data[3]) == int(vid_name[-3]) and int(data[4]) == int(vid_name[-2]) and int(data[5]) == (int(vid_name[-3]) + frame_no):
                bb_data = list(map(int, data[6:]))
                break
    if bb_data == []:
        return (frame[prev_bb[2]:prev_bb[3], prev_bb[0]:prev_bb[1]], prev_bb)
    center = [bb_data[0]+bb_data[2]//2, bb_data[1]+bb_data[3]//2]
    pad_dim = [center[0], center[0], center[1], center[1]]
    if (center[0] + max_width//2) > width:
        pad_dim[0] = center[0] - max_width//2 - \
            (max_width//2 - (width-center[0]))
        pad_dim[1] = width
    elif (center[0] - max_width//2) < 0:
        pad_dim[0] = 0
        pad_dim[1] = center[0] + max_width//2 + (max_width//2 - center[0])
    else:
        pad_dim[0] = center[0] - max_width//2
        pad_dim[1] = center[0] + max_width//2
    if (center[1] + max_height//2) > height:
        pad_dim[2] = center[1] - max_height//2 - \
            (max_height//2 - (height-center[1]))
        pad_dim[3] = height
    elif (center[1] - max_height//2) < 0:
        pad_dim[2] = 0
        pad_dim[3] = center[1] + max_height//2 + (max_height//2 - center[1])
    else:
        pad_dim[2] = center[1] - max_height//2
        pad_dim[3] = center[1] + max_height//2
    pad_dim = list(map(int, pad_dim))
    return (frame[pad_dim[2]:pad_dim[3], pad_dim[0]:pad_dim[1]], pad_dim)


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


def extract_features(files_name):
    video_label = []
    video_name = []
    for cnt, item in enumerate(files_name):
        rgb_features = []
        flow_features = []
        bb_data = []
        cap = cv2.VideoCapture(data_dir+item)
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        chk, initial_frame = cap.read()
        if chk is False:
            continue
        initial_frame, prev_bb = crop_frame(
            item, 0, initial_frame, cap.get(3), cap.get(4), [])
        initial_frame = cv2.resize(
            initial_frame, (224, 224), interpolation=cv2.INTER_AREA)
        for i in range(frame_length-1):
            chk, frame = cap.read()
            if chk is False:
                continue
            frame, prev_bb = crop_frame(
                item, i+1, frame, cap.get(3), cap.get(4), prev_bb)
            bb_data.append(prev_bb)
            image = cv2.resize(frame, (224, 224),
                              interpolation=cv2.INTER_AREA)
            rgb_features.append(compute_rgb(image))
            flow_features.append(compute_flow(image, initial_frame))
            initial_frame = image
        video_label.append(item.split("_")[0])
        video_name.append(item)
        hf_rgb = h5py.File(
            "/tmp/Data/virat_input/data_file/"+item+".h5", 'w')
        hf_flow = h5py.File(
            "/tmp/Data/virat_input/context_file/"+item+".h5", 'w')
        hf_bb = h5py.File("/tmp/Data/virat_input/bb_file/"+item+".h5", 'w')
        hf_bb.create_dataset('bb_file', data=bb_data)
        hf_rgb.create_dataset('data_file', data=rgb_features)
        hf_flow.create_dataset('context_file', data=flow_features)
        hf_rgb.close()
        hf_flow.close()
        hf_bb.close()
        if cnt % 50 == 0:
            print(cnt)
        else:
            print('.', end='')

    return video_label, video_name
