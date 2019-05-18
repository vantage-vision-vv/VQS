import os
import time
import tensorflow as tf
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getbb(img):
    try:
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img)
        lblareas = stats[1:, cv2.CC_STAT_AREA]
        imax = max(enumerate(lblareas), key=(lambda x: x[1]))[0] + 1
        return [stats[imax, cv2.CC_STAT_LEFT], stats[imax, cv2.CC_STAT_TOP], stats[imax, cv2.CC_STAT_WIDTH], stats[imax, cv2.CC_STAT_HEIGHT]]

    except:
        return None


# action localization
path = 'Demo/Data/input_attn/'
X_fcn = []
pred_fcn = []
for fi in os.listdir(path):
    data = np.load((path + fi))
    inp = data['arr_0']
    att = data['arr_1']
    label = data['arr_2']
    X_fcn.append([inp, att, label])

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(
        'Models/SegNet/segnet_model-42.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Models/SegNet/'))
    graph = tf.get_default_graph()

    #inputs_pl = graph.get_tensor_by_name('input:0')
    #output = graph.get_tensor_by_name('conv_classifier/output:0')
    #att_map_pl = graph.get_tensor_by_name('attention:0')
    #is_training = graph.get_tensor_by_name('train_bool:0')

    file_cnt = 1
    for X_temp in X_fcn:
        try:
            inputs_pass, att_pass, label_pass = X_temp
        except Exception:
            error_cnt += 1
            continue
        name_cnt = 1
        for i in range(30):
            # pred = sess.run(output, feed_dict={
            #    inputs_pl: inputs_pass[i:i+1],
            #    att_map_pl: att_pass[i:i+1],
            #    is_training: False
            # })
            # pred = np.array(np.argmax(pred, axis=-1),
            #                dtype=np.int8).reshape((224, 224))
            target = np.array(label_pass[i:i+1],
                              dtype=np.int8).reshape((224, 224))
            print(np.sum(target), end = '')
            continue
            box = getbb(target)
            if box is None:

                continue

            img = np.array(inputs_pass[i:i+1]).reshape((224, 224, 3))
            cv2.rectangle(img, (box[0], box[1]),
                          (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
            cv2.imwrite('Demo/Data/images_test/img_'+str(file_cnt) +
                        '_'+str(name_cnt)+'.png', img)
            name_cnt += 1
        file_cnt += 1
end = time.time()
