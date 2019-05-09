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

vgg_conv = VGG16(weights='imagenet',include_top= False,input_shape=(224,224,3))
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

model = models.Sequential()
model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(no_of_verbs, activation='softmax'))


train_x = []
train_y = []



def extract_image_and_label(vid):
    img = []
    lab = []
    cap = cv2.VideoCapture(video_path+vid)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(length):
        _,frame = cap.read()
        frame = cv2.resize(frame,(224,224))
        img.append(np.array(frame))
        lab_arr = np.zeros((no_of_verbs,))
        lab_arr[int(vid.split("_")[0])-1] = 1
        lab.append(lab_arr)
    return img,lab

def train():
    global model
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    model.fit(train_x,train_y,epochs = 50,batch_size=1)
    model_json = model.to_json()
    with open("verb_model.json","w") as json_file:
        json_file.write(model_json)
    model.save_weights("verb_models.h5")


def prepare_data():
    global train_x,train_y
    records = {}
    files = os.listdir(data_file)
    for item in files:
        data = np.load(data_file+item)
        if item.split("_")[0] in records.keys():
            temp = records.get(item.split("_")[0])
            temp.append(data['arr_1'])
            records[item.split("_")[0]] = temp
        else:
            records[item.split("_")[0]] = [data['arr_1']]
    for key in records.keys():
        vid_name = records.get(key)
        for vid in vid_name:
            img,lab = extract_image_and_label(vid)
            train_x.extend(img)
            train_y.extend(lab)
        

