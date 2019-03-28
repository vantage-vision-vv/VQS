import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, 'Utils')
from data_extractor_hmdb51 import Data


if __name__ == '__main__':

    data = Data()
    _, _, test = data.get_split()

    pred_test = []
    y_test = []
    # training session
    with tf.Session() as sess:
        #################################
        # load saved model
        saver = tf.train.import_meta_graph('Models/videolstm_model-42.meta')
        saver.restore(sess, tf.train.latest_checkpoint('Models/'))
        graph = tf.get_default_graph()
        Z = graph.get_tensor_by_name('input:0')
        y = graph.get_tensor_by_name('output:0')
        #################################
        # test data predictions
        for i in range(len(test)):
            Z_pass, y_target = data.get_data(test[i])
            pred = sess.run(y, feed_dict={Z: Z_pass})

            pred_test.append(np.argmax(pred))
            y_test.append(y_target)

    report = classification_report(y_true=y_test, y_pred=pred_test)
    print('Classification report for test data: ')
    print(report)