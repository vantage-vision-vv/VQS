import cv2
import os
import numpy as np
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

data_file = "/home/alpha/Work/Dataset/Virat_Ground/ggnn_input/"
video_path = "/home/alpha/Work/Dataset/Virat_Ground/Virat_Trimed/"

no_of_verbs = 12

vgg_conv = VGG16(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

model = models.Sequential()
model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(no_of_verbs, activation='softmax'))

total_samples = 0
file_names = []
files = os.listdir(data_file)
batch_size = 2
for item in files:
    vid_name = np.load(data_file+item)['arr_1'][0]
    vid_split = vid_name.split("_")
    no_of_frames = int(vid_split[-2]) - int(vid_split[-3]) + 1
    total_samples += no_of_frames
    for i in range(no_of_frames):
        file_names.append(str(i) + "_" + vid_name)


def extract_image_and_label(vid_batch):
    img =  []
    lab = []
    for vid in vid_batch:
        frame_no = int(vid.split("_")[0])
        vid_name = "_".join(vid.split("_")[1:])
        cap = cv2.VideoCapture(video_path + vid_name)
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(totalFrames):
            _ , frame = cap.read()
            if i != frame_no:
                continue
            frame = cv2.resize(frame,(224,224))
            img.append(np.array(frame))
            temp_arr = np.zeros((no_of_verbs,))
            temp_arr[int(vid_name.split("_")[0]) - 1] = 1
            lab.append(temp_arr)
    return np.array(img),np.array(lab) 


def train():
    global model
    batch_generator = MY_Generator(file_names,batch_size,total_samples)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.fit_generator(generator=batch_generator,epochs = 20,)
    model_json = model.to_json()
    with open("verb_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("verb_models.h5")



class MY_Generator():
    def __init__(self, file_names, batch_size, total_samples):
        self.file_names=file_names
        self.batch_size=batch_size
        self.total_samples=total_samples
    def __len__(self):
        return np.ceil(self.total_samples/self.batch_size)
    def __getitem__(self, idx):
        files=self.file_names[idx *
            self.batch_size: (idx + 1) * self.batch_size]
        x,y = extract_image_and_label(files)
        return x,y

train()