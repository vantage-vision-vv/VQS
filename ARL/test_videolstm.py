import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# import appropriate dataset
import sys
sys.path.insert(0, 'Utils')
from data_extractor_crime import Data


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
        out = graph.get_tensor_by_name('output:0')
        Att_t = graph.get_tensor_by_name('attention:0')
        #################################
        # test data predictions
        for i in range(len(test)):
            Z_pass, y_target = data.get_data(test[i])
            pred,att_map = sess.run([out,Att_t], feed_dict={Z: Z_pass})

            pred_test.append(np.argmax(pred))
            y_test.append(y_target)

    report = classification_report(y_true=y_test, y_pred=pred_test)
    acc = accuracy_score(y_true=y_test, y_pred=pred_test)
    print('Classification accuracy for test data: ' + str(acc))
    print('Classification report for test data: ')
    print(report)
    print(att_map)
