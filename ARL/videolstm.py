import tensorflow as tf
from tensorflow.nn import conv2d as conv
from tensorflow.math import sigmoid as sig
from tensorflow.math import tanh as tanh
from tensorflow.math import multiply as mul
from tensorflow.math import exp as exp
import numpy as np

import sys
sys.path.insert(0, '/home/alpha/Work/VQS/Utils')
from data_extractor_hmdb51 import Data


class videolstm(object):
    def __init__(self):
        self.cnn_filter_1 = (1, 1, 512, 512)
        self.cnn_filter_2 = (1, 1, 512, 512)
        self.cnn_filter_3 = (3, 3, 512, 512)
        self.cnn_filter_4 = (3, 3, 512, 512)
        self.ctx_kernel_size = (1, 1, 512, 512)
        self.ctx_transition_size = (1, 1, 512, 512)
        self.hybrid_transition_size = (1, 1, 512, 512)
        self.kernel_size = (3, 3, 512, 512)
        self.transition_size = (3, 3, 512, 512)
        self.feature_in = 512
        self.feature_out = 512
        self.ctx_in = 512
        self.ctx_out = 512
        self.atten_kernel = (3,3,512,512)
        self.atten_transition = (1,1,512,1)
        self.atten_bias = (1,7,7,512)

        self.actions = 51  # HMDB51
        self.timesteps = 30
        self.Z = tf.placeholder(
            tf.float32, shape=(self.timesteps, 7, 7, 1024))
        self.y = tf.placeholder(
            tf.float32, shape=(self.actions))

        self.lr = 0.001

        tf.set_random_seed(42)
        np.random.seed(42)
        self.network()

    def network(self):
        # Initilaizer
        self.W_cnn_1 = tf.get_variable(
            "W_cnn_1", shape=self.cnn_filter_1, initializer=tf.contrib.layers.xavier_initializer())
        self.W_cnn_2 = tf.get_variable(
            "W_cnn_2", shape=self.cnn_filter_2, initializer=tf.contrib.layers.xavier_initializer())
        self.W_cnn_3 = tf.get_variable(
            "W_cnn_3", shape=self.cnn_filter_3, initializer=tf.contrib.layers.xavier_initializer())
        self.W_cnn_4 = tf.get_variable(
            "W_cnn_4", shape=self.cnn_filter_4, initializer=tf.contrib.layers.xavier_initializer())

        # motion network
        self.W_first_xi = tf.get_variable(
            "W_first_xi", shape=self.ctx_kernel_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_xf = tf.get_variable(
            "W_first_xf", shape=self.ctx_kernel_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_xo = tf.get_variable(
            "W_first_xo", shape=self.ctx_kernel_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_xc = tf.get_variable(
            "W_first_xc", shape=self.ctx_kernel_size, initializer=tf.contrib.layers.xavier_initializer())

        self.W_first_hi = tf.get_variable(
            "W_first_hi", shape=self.ctx_transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_hf = tf.get_variable(
            "W_first_hf", shape=self.ctx_transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_ho = tf.get_variable(
            "W_first_ho", shape=self.ctx_transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_hc = tf.get_variable(
            "W_first_hc", shape=self.ctx_transition_size, initializer=tf.contrib.layers.xavier_initializer())

        self.W_first_ei = tf.get_variable(
            "W_first_ei", shape=self.hybrid_transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_ef = tf.get_variable(
            "W_first_ef", shape=self.hybrid_transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_eo = tf.get_variable(
            "W_first_eo", shape=self.hybrid_transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_first_ec = tf.get_variable(
            "W_first_ec", shape=self.hybrid_transition_size, initializer=tf.contrib.layers.xavier_initializer())

        self.b_first_i = tf.get_variable(
            "b_first_i", shape=self.ctx_out, initializer=tf.constant_initializer(value=0))
        self.b_first_f = tf.get_variable(
            "b_first_f", shape=self.ctx_out, initializer=tf.constant_initializer(value=0))
        self.b_first_o = tf.get_variable(
            "b_first_o", shape=self.ctx_out, initializer=tf.constant_initializer(value=0))
        self.b_first_c = tf.get_variable(
            "b_first_c", shape=self.ctx_out, initializer=tf.constant_initializer(value=0))

        # appearance network
        self.W_second_xi = tf.get_variable(
            "W_second_xi", shape=self.kernel_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_second_xf = tf.get_variable(
            "W_second_xf", shape=self.kernel_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_second_xo = tf.get_variable(
            "W_second_xo", shape=self.kernel_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_second_xc = tf.get_variable(
            "W_second_xc", shape=self.kernel_size, initializer=tf.contrib.layers.xavier_initializer())

        self.W_second_hi = tf.get_variable(
            "W_second_hi", shape=self.transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_second_hf = tf.get_variable(
            "W_second_hf", shape=self.transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_second_ho = tf.get_variable(
            "W_second_ho", shape=self.transition_size, initializer=tf.contrib.layers.xavier_initializer())
        self.W_second_hc = tf.get_variable(
            "W_second_hc", shape=self.transition_size, initializer=tf.contrib.layers.xavier_initializer())

        self.b_second_i = tf.get_variable(
            "b_second_i", shape=self.feature_out, initializer=tf.constant_initializer(value=0))
        self.b_second_f = tf.get_variable(
            "b_second_f", shape=self.feature_out, initializer=tf.constant_initializer(value=0))
        self.b_second_o = tf.get_variable(
            "b_second_o", shape=self.feature_out, initializer=tf.constant_initializer(value=0))
        self.b_second_c = tf.get_variable(
            "b_second_c", shape=self.feature_out, initializer=tf.constant_initializer(value=0))

        # attention cnn
        self.W_inter_xa = tf.get_variable("W_inter_xa", shape=self.atten_kernel, initializer=tf.contrib.layers.xavier_initializer())
        self.W_inter_ha = tf.get_variable("W_inter_ha", shape=self.atten_kernel, initializer=tf.contrib.layers.xavier_initializer())
        self.W_inter_z = tf.get_variable("W_inter_z", shape=self.atten_transition, initializer=tf.contrib.layers.xavier_initializer())
        self.b_inter_a = tf.get_variable("b_inter_a", shape=self.atten_bias, initializer=tf.constant_initializer(value=0))

    def clstm_forward(self, prev, inp):
        H_first_tm1, C_first_tm1, H_second_tm1, C_second_tm1, _ = prev  
        M_t = inp[:, :, :512]
        X_t = inp[:, :, 512:]
        M_t = tf.reshape(M_t, (1, 7, 7, 512))
        X_t = tf.reshape(X_t, (1, 7, 7, 512))
        # first network motion layer
        I_first_t = sig(
            tf.add(
                tf.add(
                    tf.add(
                        conv(M_t, self.W_first_xi, padding='SAME'), conv(H_first_tm1, self.W_first_hi, padding='SAME')),
                    conv(H_second_tm1, self.W_first_ei, padding='SAME')),
                self.b_first_i))
        F_first_t = sig(
            tf.add(
                tf.add(
                    tf.add(
                        conv(M_t, self.W_first_xf, padding='SAME'), conv(H_first_tm1, self.W_first_hf, padding='SAME')),
                    conv(H_second_tm1, self.W_first_ef, padding='SAME')),
                self.b_first_f))
        O_first_t = sig(
            tf.add(
                tf.add(
                    tf.add(
                        conv(M_t, self.W_first_xo, padding='SAME'), conv(H_first_tm1, self.W_first_ho, padding='SAME')),
                    conv(H_second_tm1, self.W_first_eo, padding='SAME')),
                self.b_first_o))
        G_first_t = tanh(
            tf.add(
                tf.add(
                    tf.add(
                        conv(M_t, self.W_first_xc, padding='SAME'), conv(H_first_tm1, self.W_first_hc, padding='SAME')),
                    conv(H_second_tm1, self.W_first_ec, padding='SAME')),
                self.b_first_c))
        C_first_t = tf.add(mul(F_first_t, C_first_tm1),
                           mul(I_first_t, G_first_t))
        H_first_t = mul(O_first_t, tanh(C_first_t))

        # intermediate attention cnn layer
        Z_t = conv(
            tanh(
                tf.add(
                    tf.add(
                        conv(X_t, self.W_inter_xa, padding='SAME'),
                        conv(H_first_t, self.W_inter_ha, padding='SAME')),
                    self.b_inter_a)),
            self.W_inter_z, padding='SAME')
        A_t = exp(Z_t)/np.sum(exp(Z_t))
        X_tilda_t = mul(A_t, X_t)

        # second appearance layer
        I_second_t = sig(
            tf.add(
                tf.add(
                    conv(X_tilda_t, self.W_second_xi, padding='SAME'), conv(H_second_tm1, self.W_second_hi, padding='SAME')),
                self.b_second_i))
        F_second_t = sig(
            tf.add(
                tf.add(
                    conv(X_tilda_t, self.W_second_xf, padding='SAME'), conv(H_second_tm1, self.W_second_hf, padding='SAME')),
                self.b_second_f))
        O_second_t = sig(
            tf.add(
                tf.add(
                    conv(X_tilda_t, self.W_second_xo, padding='SAME'), conv(H_second_tm1, self.W_second_ho, padding='SAME')),
                self.b_second_o))
        G_second_t = sig(
            tf.add(
                tf.add(
                    conv(X_tilda_t, self.W_second_xc, padding='SAME'), conv(H_second_tm1, self.W_second_hc, padding='SAME')),
                self.b_second_c))
        C_second_t = tf.add(mul(F_second_t, C_second_tm1),
                            mul(I_second_t, G_second_t))
        H_second_t = mul(O_second_t, tanh(C_second_t))

        return [H_first_t, C_first_t, H_second_t, C_second_t, A_t]

    def forward(self, Z):
        M_t = Z[:, :, :, :512]
        X_t = Z[:, :, :, 512:]
        # CNN_1 initialization for first lstm cell state
        cnn_1_output = conv(M_t[0:1], self.W_cnn_1, padding="SAME")

        # CNN_2 initialization for first lstm hidden state
        cnn_2_output = conv(M_t[0:1], self.W_cnn_2, padding="SAME")

        # CNN_3 initialization for second lstm cell state
        cnn_3_output = conv(X_t[0:1], self.W_cnn_3, padding="SAME")

        # CNN_4 initialization for second lstm hidden state
        cnn_4_output = conv(X_t[0:1], self.W_cnn_4, padding="SAME")

        At_t = tf.get_variable("At_t", shape=(1,7,7,1), initializer=tf.constant_initializer(value=0))
        initial_states = [cnn_2_output, cnn_1_output,
                          cnn_4_output, cnn_3_output, At_t]
        '''
        augmented_input = []
        for i in range(self.timesteps):
            augmented_input.append(tf.pack([M_t[i], X_t[i]]))
        '''
        output_states = tf.scan(
            self.clstm_forward, Z, initializer=initial_states)

        fc_input = tf.reshape(output_states[2], [1,-1])
        temp_1 = tf.layers.dense(fc_input, 1024, activation="tanh")
        temp_2 = tf.layers.dropout(temp_1, rate=0.7)
        out = tf.layers.dense(temp_2, self.actions, activation=None)
        out = tf.reshape(out, [-1])
        return out

    def train(self):
        prediction = self.forward(self.Z)
        cost = tf.losses.softmax_cross_entropy(
            onehot_labels=self.y, logits=prediction)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(cost)

        tf.get_default_graph()
        data = Data()
        train, val, _ = data.get_split()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            epoch = 0
            patience = 5
            patience_cnt = 0
            epoch_val_loss = 0
            epoch_train_loss = 0
            epoch_train_loss_prev = 0
            min_epoch_val_loss = 0

            while True:
                # training set
                for i in range(len(train)):
                    Z_pass, y_pass = data.get_data(train[i])
                    y_pass = tf.one_hot(y_pass,self.actions)
                    _, loss = sess.run([optimizer, cost], feed_dict={
                        self.Z: Z_pass,
                        self.y: y_pass
                    })
                    epoch_train_loss += loss
                epoch += 1
                epoch_train_loss = epoch_train_loss/len(train)
                # validation error
                for i in range(len(val)):
                    Z_pass, y_pass = data.get_data(val[i])
                    loss = sess.run(cost, feed_dict={
                        self.Z: Z_pass,
                        self.y: y_pass
                    })
                    epoch_val_loss += loss
                epoch_val_loss = epoch_val_loss/len(val)

                print('Epoch: '+str(epoch)+'; Training Error: ' +
                      str(epoch_train_loss)+'; Validation Error: '+str(epoch_val_loss))

                if epoch_val_loss < min_epoch_val_loss:
                    min_epoch_val_loss = epoch_val_loss
                    # save the model
                    patience_cnt = 0

                elif epoch_val_loss >= min_epoch_val_loss:
                    patience_cnt += 1

                if epoch == 3000 or (epoch_train_loss_prev - epoch_train_loss) <= 0.001 or patience_cnt == patience:
                    break

                epoch_train_loss_prev = epoch_train_loss
                epoch_train_loss = 0
                epoch_val_loss = 0


v = videolstm()
v.train()
