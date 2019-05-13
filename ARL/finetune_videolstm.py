import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from videolstm import videolstm

# import appropriate dataset
import sys
sys.path.insert(0, 'Utils')
from data_extractor_vlstm import Data


if __name__ == '__main__':

    data = Data()
    train = data.get_split("train")
    val = data.get_split("val")

    vlstm = videolstm()

    # training session
    with tf.Session() as sess:
        # load saved model
        saver = tf.train.import_meta_graph(
            'Models/VIRAT/videolstm_model-42.meta')
        saver.restore(sess, tf.train.latest_checkpoint('Models/VIRAT/'))
        graph = tf.get_default_graph()

        Z = graph.get_tensor_by_name('input:0')
        out = graph.get_tensor_by_name('output:0')
        y = tf.placeholder(tf.float32, shape=(vlstm.actions))
        optimizer = graph.get_tensor_by_name('Adam:0')


        cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=out)
        
        epoch = 0
        patience = 5
        patience_cnt = 0
        epoch_val_loss = 0
        epoch_train_loss = 0
        min_epoch_val_loss = 1e+10

        while True:
            # training set
            for i in range(len(train)):
                try:
                    Z_pass, y_pass = data.get_data(train[i], "train")
                except Exception:
                    continue

                y_ohe = np.zeros(shape=vlstm.actions)
                y_ohe[int(y_pass)-1] = 1
                _, loss = sess.run([optimizer, cost], feed_dict={
                    Z: Z_pass,
                    y: y_ohe
                })
                epoch_train_loss += loss
            epoch_train_loss = epoch_train_loss/len(train)

            # validation set
            for i in range(len(val)):
                try:
                    Z_pass, y_pass = data.get_data(val[i], "val")
                except Exception:
                    continue
                y_ohe = np.zeros(shape=vlstm.actions)
                y_ohe[int(y_pass)-1] = 1
                loss = sess.run(cost, feed_dict={
                    Z: Z_pass,
                    y: y_ohe
                })
                epoch_val_loss += loss
            epoch_val_loss = epoch_val_loss/len(val)

            print('Epoch: '+str(epoch)+'; Training Error: ' +
                  str(epoch_train_loss)+'; Validation Error: '+str(epoch_val_loss))

            if epoch_val_loss < min_epoch_val_loss:
                min_epoch_val_loss = epoch_val_loss
                ################################
                # save the model
                saver = tf.train.Saver()
                saver.save(sess, 'Models/VIRAT/videolstm_model',
                           global_step=42)
                ################################
                patience_cnt = 0

            elif epoch_val_loss >= min_epoch_val_loss:
                patience_cnt += 1

            # break conditions
            if epoch == 20 or patience_cnt == patience:
                break

            epoch_train_loss = 0
            epoch_val_loss = 0
            epoch += 1
