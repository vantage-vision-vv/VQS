from videolstm import videolstm
import tensorlow as tf
import numpy as np
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, 'Utils')
from data_extractor_hmdb51 import Data


if __name__ == '__main__':
    vlstm = videolstm()
    #################################
    # load saved model
    #################################
    prediction = vlstm.forward(vlstm.Z)

    tf.get_default_graph()
    data = Data()
    _, _, test = data.get_split()

    pred_test = []
    y_test = []
    # training session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # test data predictions
        for i in range(len(test)):
            Z_pass, y_target = data.get_data(test[i])
            pred = sess.run(prediction, feed_dict={vlstm.Z: Z_pass})

            pred_test.append(np.argmax(pred))
            y_test.append(y_target)

    report = classification_report(y_true=y_test, y_pred=pred_test)
    print('Classification report for test data: ')
    print(report)
