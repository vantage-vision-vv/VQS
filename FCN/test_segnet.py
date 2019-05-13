import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# import appropriate dataset
import sys
sys.path.insert(0, 'Utils')
from data_extractor_segnet import Data


if __name__ == '__main__':

    data = Data()
    test = data.get_split("test")

    pred_test = []
    y_test = []
    # training session
    with tf.Session() as sess:
        #################################
        # load saved model
        saver = tf.train.import_meta_graph(
            'Models/SegNet/segnet_model-42.meta')
        saver.restore(sess, tf.train.latest_checkpoint('Models/SegNet/'))
        graph = tf.get_default_graph()
        
        inputs_pl = graph.get_tensor_by_name('input:0')
        output = graph.get_tensor_by_name('output:0')
        att_map_pl = graph.get_tensor_by_name('attention:0')
        is_training = graph.get_tensor_by_name('train_bool:0')

        #################################
        # test data predictions
        for i in range(len(test)):
            try:
                inputs_pass, att_pass, label_pass = data.get_data(
                    test[i], "test")
            except Exception:
                error_cnt += 1
                continue
            for i in range(30):
                pred = sess.run(output, feed_dict={
                    inputs_pl: inputs_pass[i:i+1],
                    att_map_pl: att_pass[i:i+1],
                    is_training: False
                })
                