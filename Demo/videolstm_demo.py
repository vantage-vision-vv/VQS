import time
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# action classification
path = 'Demo/Data/input/'
X_demo = []
y_demo = []
pred_demo = []
for fi in os.listdir(path):
    Z = np.load((path + fi))
    x, y, _ = Z
    X_demo.append(x.reshape((30, 7, 7, 1024)))
    y_demo.append(y)

X_demo = np.array(X_demo).astype('float32')
y_demo = np.array(y_demo).astype('int')

with tf.Session() as sess:
    # load saved model
    saver = tf.train.import_meta_graph('Models/VIRAT/videolstm_model-42.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Models/VIRAT/'))

    graph = tf.get_default_graph()

    Z = graph.get_tensor_by_name('input:0')
    out = graph.get_tensor_by_name('output:0')

    for X_temp in X_demo:
        pred = sess.run(out, feed_dict={Z: X_temp})
        pred_demo.append(np.argmax(pred)+1)

