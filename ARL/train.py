from videolstm import videolstm
import tensorflow as tf
import numpy as np

# import appropriate dataset
import sys
sys.path.insert(0, 'Utils')
from data_extractor_arl import Data


if __name__ == '__main__':
    vlstm = videolstm()

    # prediction vecotr cost function and optimizer
    prediction = vlstm.forward(vlstm.Z)
    cost = tf.losses.softmax_cross_entropy(
        onehot_labels=vlstm.y, logits=prediction)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=vlstm.lr).minimize(cost)

    tf.get_default_graph()
    data = Data()
    train = data.get_split("train")
    val = data.get_split("val")

    # training session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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
                    vlstm.Z: Z_pass,
                    vlstm.y: y_ohe
                })
                epoch_train_loss += loss
            epoch += 1
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
                    vlstm.Z: Z_pass,
                    vlstm.y: y_ohe
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
                saver.save(sess, 'Models/ARL/VIRAT/videolstm_model',
                           global_step=42)
                ################################
                patience_cnt = 0

            elif epoch_val_loss >= min_epoch_val_loss:
                patience_cnt += 1

            # break conditions
            if epoch == 1000 or patience_cnt == patience:
                break

            epoch_train_loss = 0
            epoch_val_loss = 0
