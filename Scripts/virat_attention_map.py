import numpy as np
import tensorflow as tf
import cv2

vid_path = "/tmp/hmdb51/hmdb/"
data_path = "/tmp/Data/hmdb_input/"
map_path = "./Data/"


def run_vlstm(inp):
    inp = np.swapaxes(inp, 1, 2)
    inp = np.swapaxes(inp, 2, 3)
    print(inp.shape)
    att_map = sess.run([Att_t], feed_dict={Z: inp})
    return att_map


def get_label_image(bb):
    bb = list(map(float, bb))
    bb = list(map(int, bb))
    label = np.zeros((224, 224))
    label[bb[0]:bb[2]+bb[0], bb[1]:bb[3]+bb[1]] = 1
    return label


def store_attn(p):
    with open(map_path+"map_"+p+".txt", "r") as f:
        for cnt, line in enumerate(f):
            data = line.strip().split(" ")
            input_data = np.load(data_path + p + "/" + data[0] + ".npy")
            frame_data = list(map(int, data[1].split(",")))
            vid_name = "_".join(x for x in data[0].split("_")[:-1])
            cap = cv2.VideoCapture(vid_path + vid_name)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_data = []
            label_data = []
            _, _ = cap.read()
            for i in range(length-1):
                ret, frame = cap.read()
                if i in frame_data:
                    gray = cv2.resize(frame, (224, 224))
                    img_data.append(gray)
                    label_data.append(get_label_image(
                        input_data[2][len(img_data) - 1]))
            attn_data = run_vlstm(input_data[0])
            np.savez(data_path + p + "_attn/" + str(cnt) + ".npy", np.array(img_data).reshape((10, 224, 224, 3)),
                     np.array(attn_data).reshape((10, 7, 7, 1)), np.array(label_data).reshape((10, 224, 224, 1)))


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('Models/VIRAT/videolstm_model-42.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Models/VIRAT/'))
    graph = tf.get_default_graph()
    Z = graph.get_tensor_by_name('input:0')
    Att_t = graph.get_tensor_by_name('attention:0')
    store_attn("train")
    store_attn("val")
    store_attn("test")
