import cv2
import os
import numpy as np
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import Sequence

data_file = "/tmp/Data/ggnn_input/"
video_path = "/tmp/Virat_Trimed/"

no_of_noun = 5

vgg_conv = VGG16(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

model = models.Sequential()
model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(no_of_noun, activation='sigmoid'))

total_samples = 0
file_names = []
labels = []
files = os.listdir(data_file)
batch_size = 2
for item in files:
    vid_name = np.load(data_file+item)['arr_1'][0]
    objects = np.load(data_file+item)['arr_0'].tolist()
    objects = list(map(int, objects))
    print(objects)

    vid_split = vid_name.split("_")
    no_of_frames = int(vid_split[-2]) - int(vid_split[-3]) + 1
    for i in range(no_of_frames):
        if i % 15 == 0:
            file_names.append(str(i) + "_" + vid_name)
            labels.append(objects)
total_samples = len(file_names)


def extract_image_and_label(vid_batch, label):
    img = []
    lab = []
    for cnt, vid in enumerate(vid_batch):
        frame_no = int(vid.split("_")[0])
        vid_name = "_".join(vid.split("_")[1:])
        cap = cv2.VideoCapture(video_path + vid_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        chk, frame = cap.read()
        if chk == False:
            frame = np.zeros((224, 224, 3))
        frame = cv2.resize(frame, (224, 224))
        img.append(np.array(frame))
        temp_arr = np.zeros((no_of_noun,))
        for index in label[cnt]:
            temp_arr[index - 1] = 1
        lab.append(temp_arr)
        cap.release()
    return np.array(img), np.array(lab)


def train():
    global model
    batch_generator = MY_Generator(
        file_names, labels, batch_size, total_samples)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.fit_generator(generator=batch_generator, epochs=1)
    model_json = model.to_json()
    with open("noun_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("noun_models.h5")


class MY_Generator(Sequence):
    def __init__(self, file_names, labels, batch_size, total_samples):
        self.file_names = file_names
        self.labels = labels
        self.batch_size = batch_size
        self.total_samples = total_samples

    def __len__(self):
        return int(np.ceil(self.total_samples/self.batch_size))

    def __getitem__(self, idx):
        files = self.file_names[idx *
                                self.batch_size: (idx + 1) * self.batch_size]
        lab = self.labels[idx *
                          self.batch_size: (idx + 1) * self.batch_size]
        x, y = extract_image_and_label(files, lab)
        return x, y


train()
