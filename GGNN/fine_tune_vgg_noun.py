import cv2
import os
import numpy as np
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

data_file = "/home/alpha/Work/Dataset/Virat_Ground/ggnn_input/"
video_path = "/home/alpha/Work/Dataset/Virat_Ground/Virat_Trimed/"

no_of_noun = 5

vgg_conv = VGG16(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

model = models.Sequential
model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(no_of_noun, activation='sigmoid'))


train_x = []
train_y = []


def extract_image_and_label(vid, objects):
    img = []
    lab = []
    cap = cv2.VideoCapture(video_path+vid[0])
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(length):
        _, frame = cap.read()
        frame = cv2.resize(frame, (224, 224))
        img.append(np.array(frame))
        lab_arr = np.zeros((no_of_noun,))
        for item in objects:
            lab_arr[item-1] = 1low(curr, prev):
        lab.append(lab_arr)
    return img, lab


def train():
    global model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.fit(train_x, train_y, epochs=50, batch_size=1)
    model_json = model.to_json()
    with open("noun_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("noun_models.h5")


def prepare_data():
    global train_x, train_y
    records = {}
    files = os.listdir(data_file)
    for item in files:
        data = np.load(data_file+item)
        img, lab = extract_image_and_label(data['arr_1'], data['arr_0'])


low(curr, prev): low(curr, prev):
prepare_data()
train()
