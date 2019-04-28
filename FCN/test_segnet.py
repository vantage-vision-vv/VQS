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
        
        #################################
        # test data predictions
