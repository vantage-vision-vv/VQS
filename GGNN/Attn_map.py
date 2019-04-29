import numpy as np
import tensorflow as tf
import cv2

vid_path = "/tmp/hmdb51/hmdb/"
data_path = "/tmp/Data/hmdb_input/"
map_path = "./"


def run_vlstm(inp):
    att_map = sess.run([Att_t], feed_dict={Z: inp})
    return att_map

def get_label_image(bb_data):
    label = np.zeros((224,224))
    return label[bb_data[0]:bb_data[2]+bb_data[0],bb_data[1]:bb_data[3]+bb_data[1]] = 1



def store_attn(p):
    with open(map_path+"map_"+p+".txt","r") as f:
        for cnt,line in enumerate(f):
            data = line.strip().split(" ")
            input_data = np.load(data_path + p + "/" + data[1] + ".npy")
            frame_data = map(int,data[2].split(","))
            cap = cv2.VideoCapture(vid_path + data[0] + ".avi")
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_data = []
            attn_data = []
            label_data = []
            for i in range(length):
                ret, frame = cap.read()
                if i in frame_data:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    img_data.append(gray)
                    attn_data.append(run_vlstm(input_data[0][len(img_data) - 1]))
                    label_data.append(get_label_image(input_data[2][len(img_data) - 1]))
            
            final_data = [np.array(img_data),np.array(attn_data),np.array(label_data)]
            np.save(data_path + p +"_attn/" + cnt + ".npy",final_data)                        



with tf.Session() as sess:
    saver = tf.train.import_meta_graph('Models/HMDB51/videolstm_model-42.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Models/HMDB51/'))
    graph = tf.get_default_graph()
    Z = graph.get_tensor_by_name('input:0')
    Att_t = graph.get_tensor_by_name('attention:0')
    store_attn("train")
    store_attn("val")
    store_attn("test")














