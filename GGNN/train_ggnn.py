import cv2
import os
import numpy as np
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import Sequence
from keras.models import model_from_json, Model
from keras import backend as K
import tensorflow as tf
from GGNN import *

# Loading Data
#########################################################################################
data_file = "/home/alpha/Work/Dataset/Virat_Ground/ggnn_input/"
video_path = "/home/alpha/Work/Dataset/Virat_Ground/Virat_Trimed/"
out_path = "/home/alpha/Work/Dataset/Virat_Ground/ggnn_data/"

file_train = []
lab_train = []
file_val = []
lab_val = []
file_test = []
lab_test = []
files = os.listdir(data_file)
records = {}


def prepare_data(files):
    file_names = []
    labels = []
    for item in files:
        vid_name = np.load(data_file+item)['arr_1'][0]
        objects = np.load(data_file+item)['arr_0'].tolist()
        objects = list(map(int, objects))
        vid_split = vid_name.split("_")
        no_of_frames = int(vid_split[-2]) - int(vid_split[-3]) + 1
        for i in range(no_of_frames):
            if i % 15 == 0:
                file_names.append(str(i) + "_" + vid_name)
                labels.append(objects)
    return file_names, labels


for item in files:
    if item.split("_")[0] in records.keys():
        temp = records.get(item.split("_")[0])
        temp.append(item)
        records[item.split("_")[0]] = temp
    else:
        records[item.split("_")[0]] = [item]


for key in records.keys():
    file_train.extend(records.get(key)[:int(0.6*len(records.get(key)))])
    file_val.extend(records.get(
        key)[int(0.6*len(records.get(key))):int(0.8*len(records.get(key)))])
    file_test.extend(records.get(key)[int(0.8*len(records.get(key))):])


file_train, lab_train = prepare_data(file_train)
file_val, lab_val = prepare_data(file_val)
file_test, lab_test = prepare_data(file_test)

##########################################################################################
# Restoring model
##########################################################################################
json_file_verb = open("./Models/GGNN/verb_model.json", 'r')
json_file_noun = open("./Models/GGNN/noun_model.json", 'r')

model_json_verb = json_file_verb.read()
model_json_noun = json_file_noun.read()

json_file_verb.close()
json_file_noun.close()

verb_model = model_from_json(model_json_verb)
noun_model = model_from_json(model_json_noun)

verb_model.load_weights("./Models/GGNN/verb_models.h5")
noun_model.load_weights("./Models/GGNN/noun_models.h5")

verb_rep = Model(inputs=verb_model.input, outputs=verb_model.layers[-2].output)
noun_rep = Model(inputs=noun_model.input, outputs=noun_model.layers[-2].output)
############################################################################################

mapping = {
    1: [[1], [4], [2, 3], [0], [0], [0], [0]],
    2: [[1], [4], [0], [2, 3], [0], [0], [0]],
    3: [[1], [0], [0], [0], [2, 3], [0], [0]],
    4: [[1], [0], [0], [0], [2, 3], [0], [0]],
    5: [[1], [0], [0], [0], [0], [2, 3], [0]],
    6: [[1], [0], [0], [0], [0], [0], [2, 3]],
    7: [[1], [0], [0], [0], [0], [0], [0]],
    8: [[1], [0], [0], [0], [0], [0], [0]],
    9: [[1], [4], [0], [0], [0], [0], [0]],
    10: [[1], [0], [0], [0], [0], [0], [0]],
    11: [[1], [0], [0], [0], [0], [0], [0]],
    12: [[1], [0], [0], [0], [0], [0], [0]],
}


def extract_roles(lab, verb):
    noun_enc = np.zeros((no_of_roles, num_nouns))
    collec = mapping.get(verb)
    for item in lab:
        for index, rule in enumerate(collec):
            if item in rule:
                noun_enc[index][item-1] = 1.0

    return noun_enc.astype('float32')


def extract_data(name, lab):
    frame_no = int(name.split("_")[0])
    vid_name = "_".join(name.split("_")[1:])
    cap = cv2.VideoCapture(video_path + vid_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    chk, frame = cap.read()
    if chk == False:
        frame = np.zeros((224, 224, 3))
    frame = cv2.resize(frame, (224, 224))
    verb_vec = verb_rep.predict(frame.reshape((1, 224, 224, 3)))
    noun_vec = noun_rep.predict(frame.reshape((1, 224, 224, 3)))
    verb_enc = np.zeros((num_verbs))
    verb_enc[int(vid_name.split("_")[0]) - 1] = 1
    noun_enc = extract_roles(lab, int(vid_name.split("_")[0]))
    return verb_vec, noun_vec, verb_enc.reshape((1, 12)), noun_enc


def train():
    model = GGNN()
    model.initialize()

    cost = tf.matmul(model.verb_encoding_pl, tf.log(model.prob_verb))
    for i in range(no_of_roles):
        cost = tf.add(cost, tf.matmul(
            model.noun_encoding_pl[i:i+1, :], tf.log(model.prob_role.get(i))))

    optimizer = tf.train.AdamOptimizer(
        learning_rate=model.learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch = 0
        patience = 5
        patience_cnt = 0
        epoch_val_loss = 0
        epoch_train_loss = 0
        min_epoch_val_loss = 1e+10

        while True:
            # training set
            for i in range(len(file_train)):
                verb_vec, noun_vec, verb_enc, noun_enc = extract_data(
                    file_train[i], lab_train[i])
                _, loss = sess.run([optimizer, cost], feed_dict={
                    model.verb_pl: verb_vec,
                    model.noun_pl: noun_vec,
                    model.verb_encoding_pl: verb_enc,
                    model.noun_encoding_pl: noun_enc
                })

                epoch_train_loss += loss
            epoch_train_loss = epoch_train_loss/len(file_train)

            # validation set
            for i in range(len(file_val)):
                verb_vec, noun_vec, verb_enc, noun_enc = extract_data(
                    file_val[i], lab_val[i])
                loss = sess.run(cost, feed_dict={
                    model.verb_pl: verb_vec,
                    model.noun_pl: noun_vec,
                    model.verb_encoding_pl: verb_enc,
                    model.noun_encoding_pl: noun_enc
                })

                epoch_val_loss += loss
            epoch_val_loss = epoch_val_loss/len(file_val)

            print('Epoch: '+str(epoch)+'; Training Error: ' +
                  str(epoch_train_loss)+'; Validation Error: '+str(epoch_val_loss))

            if epoch_val_loss < min_epoch_val_loss:
                min_epoch_val_loss = epoch_val_loss
                ################################
                # save the model
                saver = tf.train.Saver()
                saver.save(sess, 'Models/GGNN/GGNN_model', global_step=42)
                ################################
                patience_cnt = 0

            elif epoch_val_loss >= min_epoch_val_loss:
                patience_cnt += 1

            # break conditions
            if epoch == 1000 or patience_cnt == patience:
                break

            epoch_train_loss = 0
            epoch_val_loss = 0
            epoch += 1


def save_data():
    for index in range(len(file_train)):
        verb_vec, noun_vec, verb_enc, noun_enc = extract_data(
            file_train[index], lab_train[index])
        np.savez(out_path+"train/" +
                 file_train[index], verb_vec, noun_vec, verb_enc, noun_enc)
    for index in range(len(file_val)):
        verb_vec, noun_vec, verb_enc, noun_enc = extract_data(
            file_val[index], lab_val[index])
        np.savez(out_path+"val/"+file_train[index],
                 verb_vec, noun_vec, verb_enc, noun_enc)
    for index in range(len(file_test)):
        verb_vec, noun_vec, verb_enc, noun_enc = extract_data(
            file_test[index], lab_test[index])
        np.savez(out_path+"test/" +
                 file_train[index], verb_vec, noun_vec, verb_enc, noun_enc)


save_data()
# train()
