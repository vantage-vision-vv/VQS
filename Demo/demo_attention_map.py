import numpy as np
import tensorflow as tf
import cv2
import h5py

vid_path = "Demo/trim_videos/"
data_path = "Demo/Data/input/"
out_path = "Demo/Data/input"
map_path = "Demo/map_demo_virat.txt"
bb_path = "Demo/Data/bb_file/"


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
    label[bb[0]:bb[2]+bb[0], bb[1]:bb[3]+bb[1]] = 1
    return cv2.resize(label, (224, 224))


def store_attn():
    filter_file = []
    with open(map_path, "r") as f:
        for cnt, line in enumerate(f):
            data = line.strip().split(" ")
            if (data[0] + ".npy") not in filter_file:
                continue
            vid_name = "_".join(x for x in data[0].split("_")[:-1])
            input_data = np.load(data_path + data[0] + ".npy")
            #bb_data = np.load(bb_path + vid_name + ".h5")
            bb_file = h5py.File(bb_path + vid_name + ".h5", 'r')
            bb_data = np.array(bb_file.get('bb_file'))
            bb_file.close()

            frame_data = list(map(int, data[1].split(",")))
            cap = cv2.VideoCapture(vid_path + vid_name)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_data = []
            label_data = []
            _, _ = cap.read()

            crop_cnt = 0
            for i in range(length-1):
                ret, frame = cap.read()
                if i in frame_data:
                    frame = frame[bb_data[i][2]:bb_data[i]
                                  [3], bb_data[i][0]:bb_data[i][1]]
                    origin_cropped = [bb_data[i][0], bb_data[i][2]]
                    bb_original = input_data[2][crop_cnt]
                    crop_cnt += 1
                    bb_cropped = [bb_original[0]-origin_cropped[0], bb_original[1] -
                                  origin_cropped[1], bb_original[2], bb_original[3]]

                    target = get_label_image(bb_cropped, frame.shape)
                    frame = cv2.resize(frame, (224, 224))
                    img_data.append(frame)
                    label_data.append(target)

            attn_data = run_vlstm(input_data[0])
            np.savez(out_path + "_attn/" + str(cnt) + ".npy", np.array(img_data).reshape((30, 224, 224, 3)),
                     np.array(attn_data).reshape((30, 7, 7, 1)), np.array(label_data).reshape((30, 224, 224, 1)))


if __name__ == "__main__":
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            'Models/VIRAT/videolstm_model-42.meta')
        saver.restore(sess, tf.train.latest_checkpoint('Models/VIRAT/'))
        graph = tf.get_default_graph()
        Z = graph.get_tensor_by_name('input:0')
        Att_t = graph.get_tensor_by_name('attention:0')
        store_attn()