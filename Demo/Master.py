import os
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


start = time.time()
# action classification
path = 'Demo/Data/input/'
X_demo = []
y_demo = []
att_demo = []
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
    Att_t = graph.get_tensor_by_name('attention:0')
    
    for X_temp in X_demo:
        pred, att = sess.run([out, Att_t], feed_dict={Z: X_temp})
        pred_demo.append(np.argmax(pred)+1)
        att_demo.append(att)
    
    report = classification_report(y_true=y_demo, y_pred=pred_demo)
    acc = accuracy_score(y_true=y_demo, y_pred=pred_demo)
    print('Classification accuracy for test data: ' + str(acc))
    print('Classification report for test data: ')
    print(report)
