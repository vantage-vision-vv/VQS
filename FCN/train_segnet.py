from fcn_segnet import segnet
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# import appropriate dataset
import sys
sys.path.insert(0, 'Utils')
from data_extractor_segnet import Data


def cal_cost(logits, labels, number_class):
    label_flatten = tf.to_int64(tf.reshape(labels, [-1]))
    logits_reshape = tf.reshape(logits, [-1, number_class])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flatten, logits=logits_reshape,
                                                                   name='normal_cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)


def plot_loss(train_loss, val_loss, epoch_count):
    x = np.linspace(1, epoch_count, epoch_count)
    plt.plot(x, train_loss, 'b', label="train")
    plt.plot(x, val_loss, 'r', label="val")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    fcn = segnet()

    cost, accuracy, prediction = cal_cost(
        logits=fcn.logits, labels=fcn.labels_pl, number_class=fcn.num_classes)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=fcn.learning_rate).minimize(cost)

    tf.get_default_graph()
    data = Data()
    train = data.get_split("train")
    val = data.get_split("val")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        epoch = 1
        patience = 5
        patience_cnt = 0
        epoch_val_loss = 0
        epoch_train_loss = 0
        min_epoch_val_loss = 1e+10
        history = []

        while True:
            error_cnt = 0
            # training set
            for i in range(len(train)):
                try:
                    inputs_pass, att_pass, label_pass = data.get_data(
                        train[i], "train")
                except Exception:
                    error_cnt += 1
                    continue
                for i in range(30):    
                    _, loss = sess.run([optimizer, cost], feed_dict={
                        fcn.inputs_pl: inputs_pass[i:i+1],
                        fcn.att_map_pl: att_pass[i:i+1],
                        fcn.labels_pl: label_pass[i:i+1],
                        fcn.is_training: True
                    })
                    epoch_train_loss += loss
            epoch_train_loss = epoch_train_loss/((len(train)*30 - error_cnt)*30)

            error_cnt = 0
            # validation set
            for i in range(len(val)):
                try:
                    inputs_pass, att_pass, label_pass = data.get_data(
                        val[i], "val")
                except Exception:
                    error_cnt += 1
                    continue
                for i in range(30):
                    loss = sess.run(cost, feed_dict={
                        fcn.inputs_pl: inputs_pass[i:i+1],
                        fcn.att_map_pl: att_pass[i:i+1],
                        fcn.labels_pl: label_pass[i:i+1],
                        fcn.is_training: False
                    })
                    epoch_val_loss += loss
            epoch_val_loss = epoch_val_loss/((len(val) - error_cnt)*30)

            history.append([epoch_train_loss, epoch_val_loss])
            # print present stats
            print('Epoch: '+str(epoch)+'; Training Error: ' +
                  str(epoch_train_loss)+'; Validation Error: '+str(epoch_val_loss))

            if epoch_val_loss < min_epoch_val_loss:
                min_epoch_val_loss = epoch_val_loss
                ################################
                # save the model
                saver = tf.train.Saver()
                saver.save(sess, 'Models/SegNet/segnet_model', global_step=42)
                ################################
                patience_cnt = 0

            elif epoch_val_loss >= min_epoch_val_loss:
                patience_cnt += 1

            # break conditions
            if epoch == 1000 or patience_cnt == patience:
                history = np.array(history)
                np.save("./Hist.npy", history)
                plot_loss(history[:, 0], history[:, 1], epoch)
                break

            epoch_train_loss = 0
            epoch_val_loss = 0
            epoch += 1
