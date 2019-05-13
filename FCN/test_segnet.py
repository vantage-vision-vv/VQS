import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import average_precision_score

# import appropriate dataset
import sys
sys.path.insert(0, 'Utils')
from data_extractor_segnet import Data


def getbb(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    lblareas = stats[1:, cv2.CC_STAT_AREA]
    imax = max(enumerate(lblareas), key=(lambda x: x[1]))[0] + 1
    return [stats[imax, cv2.CC_STAT_LEFT], stats[imax, cv2.CC_STAT_TOP], stats[imax, cv2.CC_STAT_WIDTH], stats[imax, cv2.CC_STAT_HEIGHT]]

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

if __name__ == '__main__':

    data = Data()
    test = data.get_split("test")

    iou_score = []
    # training session
    with tf.Session() as sess:
        #################################
        # load saved model
        saver = tf.train.import_meta_graph(
            'Models/SegNet/segnet_model-42.meta')
        saver.restore(sess, tf.train.latest_checkpoint('Models/SegNet/'))
        graph = tf.get_default_graph()

        inputs_pl = graph.get_tensor_by_name('input:0')
        output = graph.get_tensor_by_name('conv_classifier/output:0')
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
                try:
                    pred = np.array(np.max(pred, axis=-1), dtype=np.int8).reshape((-1))
                    target = np.array(label_pass[i], dtype=np.int8).reshape((-1))
                    #boxA = getbb(target)
                    #boxB = getbb(pred)
                    #iou = bb_intersection_over_union(boxA, boxB)
                    #if not np.isnan(iou):
                    #    iou_score.append(iou)
                    # pred_test.append(np.argmax(pred,axis=-1))
                    # y_test.append(label_pass[i])
                    
                    loss = average_precision_score(y_score=pred, y_true=target)
                    iou_score.append(loss)
                except:
                    pass
        print('map: ' + str(np.mean(iou_score)))