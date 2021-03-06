import numpy as np
import tensorflow as tf
import cv2
import h5py

vid_path = "/tmp/Virat_Trimed/"
data_path = "/tmp/Data/virat_input/"
map_path = "Data/"
bb_path = "/tmp/Data/virat_input/bb_file/"


def run_vlstm(inp):
    inp = np.swapaxes(inp, 1, 2)
    inp = np.swapaxes(inp, 2, 3)
    att_map = sess.run([Att_t], feed_dict={Z: inp})
    return att_map


def get_label_image(bb, shape):
    x,y,_ = shape
    bb = list(map(float, bb))
    bb = list(map(int, bb))
    label = np.zeros((x,y))
    label[bb[1]:bb[3]+bb[1], bb[0]:bb[2]+bb[0]] = 1
    return cv2.resize(label, (224, 224)).reshape((224,224,1))


def store_attn(p):
    filter_file = []
    with open(map_path + "virat_" + p + ".txt",'r') as myfile:
        for line in myfile:
            filter_file.append(line.strip())
    with open(map_path+"map_virat_"+p+".txt", "r") as f:
        for cnt, line in enumerate(f):
            data = line.strip().split(" ")
            if (data[0] + ".npy") not in filter_file:
                continue
            vid_name = "_".join(x for x in data[0].split("_")[:-1])
            input_data = np.load(data_path + p + "/" + data[0] + ".npy")
            bb_file = h5py.File(bb_path + vid_name + ".h5", 'r')
            bb_data = np.array(bb_file.get('bb_file'))
            bb_file.close()
            anno_file = "_".join(x for x in vid_name.split('_')[1:-4]) + ".viratdata.events.txt"
            frame_data = list(map(int, data[1].split(",")))
            temp = vid_name.split("_")
            start,end,key = temp[-3],temp[-2],temp[0]
            bb_focus = []
            print(start,end,key)
            with open(anno_path + anno_file,'r') as f:
                for bb_line in f:
                    bb_line = bb_line.strip().split(" ")
                    #print(int(bb_line[1]),int(bb_line[5]),(int(bb_line[1]) == int(key)),(int(bb_line[5]) > int(start)),(int(bb_line[5]) <= int(end)))
                    if (int(bb_line[1]) == int(key)) and (int(bb_line[5]) > int(start)) and (int(bb_line[5]) <= int(end)):
                        bb_focus.append(list(map(int,bb_line[6:])))
            cap = cv2.VideoCapture(vid_path + vid_name)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_data = []
            label_data = []
            _, _ = cap.read()
            for i in range(length-1):
                ret, frame = cap.read()
                if i in frame_data:
                    frame = frame[bb_data[i][2]:bb_data[i]
                                  [3], bb_data[i][0]:bb_data[i][1]]
                    origin_cropped = [bb_data[i][0], bb_data[i][2]]
                    bb_original = bb_focus[i]
                    bb_cropped = [bb_original[0]-origin_cropped[0], bb_original[1] -
                                  origin_cropped[1], bb_original[2], bb_original[3]]
                    target = get_label_image(bb_cropped, frame.shape)
                    frame = cv2.resize(frame, (224, 224))
                    img_data.append(frame)
                    label_data.append(target)

            attn_data = run_vlstm(input_data[0])
            np.savez(data_path + p + "_attn/" + str(cnt) + ".npy", np.array(img_data).reshape((30, 224, 224, 3)),
                     np.array(attn_data).reshape((30, 7, 7, 1)), np.array(label_data).reshape((30, 224, 224, 1)))


if __name__ == "__main__":
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            'Models/VIRAT/videolstm_model-42.meta')
        saver.restore(sess, tf.train.latest_checkpoint('Models/VIRAT/'))
        graph = tf.get_default_graph()
        Z = graph.get_tensor_by_name('input:0')
        Att_t = graph.get_tensor_by_name('attention:0')
        store_attn("train")
        store_attn("val")
        store_attn("test")
