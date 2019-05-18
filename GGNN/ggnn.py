import tensorflow as tf
import numpy as np

num_verbs = 12  # V
num_nouns = 5  # N
time_steps = 10  # T
no_of_roles = 7  # R


hidden_state_dim = 1024  # H
input_feature_dim = 1024  # I


class GGNN():
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.placeholders = {}
        self.weights = {}
        self.role_nodes = {}
        self.biases = {}
        self.embeddings = {}
        self.transform = {}
        self.prob_role = {}
        self.learning_rate = 1e-5

    def One_hot_encoder(self, total, pos):
        tmp = np.zeros((total, 1))
        tmp[pos] = 1
        return tf.convert_to_tensor(tmp)

    def Rnn(self):
        X_total = tf.get_variable("X_total_temp", shape=[hidden_state_dim, 1])

        for i in range(no_of_roles):
            X_total = tf.add(
                tf.matmul(self.weights['message'], self.role_nodes[i]), X_total)  # H * 1
        X_total = tf.add(
            tf.matmul(self.weights['message'], self.verb_node), X_total)
        X_total = tf.add(self.biases['message'], X_total)

        for i in range(no_of_roles):
            X_temp = tf.subtract(X_total, tf.matmul(
                self.weights['message'], self.role_nodes[i]))
            Z_t = tf.nn.softmax(tf.add(tf.add(tf.matmul(self.weights['W_z'], X_temp), tf.matmul(
                self.weights['U_z'], self.role_nodes[i])), self.biases['B_z']))  # H * 1
            R_t = tf.nn.softmax(tf.add(tf.add(tf.matmul(self.weights['W_r'], X_temp), tf.matmul(
                self.weights['U_r'], self.role_nodes[i])), self.biases['B_r']))
            H_t = tf.nn.tanh(tf.add(tf.add(tf.matmul(self.weights['W_h'], X_temp), tf.matmul(
                self.weights['U_h'], tf.multiply(R_t, self.role_nodes[i]))), self.biases['B_h']))
            self.role_nodes[i] = tf.add(tf.multiply(tf.subtract(
                tf.ones_like(Z_t), Z_t), self.role_nodes[i]), tf.multiply(Z_t, H_t))

        X_temp = tf.subtract(X_total, tf.matmul(
            self.weights['message'], self.verb_node))
        Z_t = tf.nn.softmax(tf.add(tf.add(tf.matmul(self.weights['W_z'], X_temp), tf.matmul(
            self.weights['U_z'], self.verb_node)), self.biases['B_z']))
        R_t = tf.nn.softmax(tf.add(tf.add(tf.matmul(self.weights['W_r'], X_temp), tf.matmul(
            self.weights['U_r'], self.verb_node)), self.biases['B_r']))
        H_t = tf.nn.tanh(tf.add(tf.add(tf.matmul(self.weights['W_h'], X_temp), tf.matmul(
            self.weights['U_h'], tf.multiply(R_t, self.verb_node))), self.biases['B_h']))
        self.verb_node = tf.add(tf.multiply(tf.subtract(
            tf.ones_like(Z_t), Z_t), self.verb_node), tf.multiply(Z_t, H_t))

    def model(self, v, r, verb_encoding):  # verb_encoding
        self.verb_node = tf.nn.relu(
            tf.matmul(self.transform['verb'], tf.transpose(self.verb_pl)))  # H * None

        for i in range(no_of_roles):
            self.role_nodes[i] = tf.nn.relu(tf.multiply(tf.multiply(tf.matmul(self.transform['noun'], tf.transpose(self.noun_pl)), tf.matmul(
                self.embeddings['noun'], self.One_hot_encoder(no_of_roles, i))), tf.matmul(self.embeddings['verb'], tf.transpose(self.verb_encoding_pl))))  # H * 1

        for _ in range(time_steps):  # Try scan
            self.Rnn()
        self.prob_verb = tf.nn.softmax(
            tf.add(tf.matmul(self.weights['W_h_v'], self.verb_node), self.biases['B_h_v']))  # V * 1

        for i in range(no_of_roles):
            self.prob_role[i] = tf.nn.softmax(tf.add(
                tf.matmul(self.weights['W_h_n'], self.role_nodes[i]), self.biases['B_h_n']))  # N * 1
        '''
		for i in range(no_of_roles):
            self.loss = tf.add(self.loss, tf.matmul(tf.transpose(
                self.noun_encoding_pl[i]), tf.log(self.prob_role[i])))
        temp = tf.constant([no_of_roles])
        self.loss = tf.divide(self.loss, temp)
        self.loss = tf.add(self.loss, tf.matmul(tf.transpose(
            self.verb_encoding_pl), tf.log(self.prob_verb)))
        '''

    def initialize(self):
        self.verb_pl = tf.placeholder(
            "float32", [None, input_feature_dim])  # I
        self.noun_pl = tf.placeholder(
            "float32", [None, input_feature_dim])  # I
        self.verb_encoding_pl = tf.placeholder(
            "float32", [1, num_verbs])  # V
        self.noun_encoding_pl = tf.placeholder(
            "float32", [no_of_roles, num_nouns])  # R x N

        for i in range(no_of_roles):
            var_name = "hidden_state_" + str(i)
            prob_var_name = "prob_" + str(i)
            self.role_nodes[i] = tf.get_variable(
                var_name, shape=[hidden_state_dim, 1])  # H * 1
            self.prob_role[i] = tf.get_variable(
                prob_var_name, shape=[1, num_nouns])  # 1 * N
        self.prob_verb = tf.get_variable(
            "prob_verb", shape=[1, num_verbs])  # 1 * V
        self.verb_node = tf.get_variable(
            "hidden_state_verb", shape=[hidden_state_dim, 1])		# H * 1

        self.embeddings['verb'] = tf.get_variable("verb_embedding", shape=[
            hidden_state_dim, num_verbs], initializer=tf.contrib.layers.xavier_initializer())  # H x V
        self.embeddings['noun'] = tf.get_variable("noun_embedding", shape=[
            hidden_state_dim, no_of_roles], initializer=tf.contrib.layers.xavier_initializer())  # H x R

        self.transform['verb'] = tf.get_variable("verb_transform", shape=[
            hidden_state_dim, input_feature_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x I
        self.transform['noun'] = tf.get_variable("noun_transform", shape=[
            hidden_state_dim, input_feature_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x I

        self.weights['message'] = tf.get_variable("message", shape=[
            hidden_state_dim, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x H
        self.weights['W_z'] = tf.get_variable("W_z", shape=[
            hidden_state_dim, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x H
        self.weights['W_r'] = tf.get_variable("W_r", shape=[
            hidden_state_dim, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x H
        self.weights['W_h'] = tf.get_variable("W_h", shape=[
            hidden_state_dim, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x H
        self.weights['U_z'] = tf.get_variable("U_z", shape=[
            hidden_state_dim, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x H
        self.weights['U_r'] = tf.get_variable("U_r", shape=[
            hidden_state_dim, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x H
        self.weights['U_h'] = tf.get_variable("U_h", shape=[
            hidden_state_dim, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())  # H x H

        self.biases['B_z'] = tf.get_variable("B_z", shape=[
            hidden_state_dim, 1], initializer=tf.contrib.layers.xavier_initializer())  # H x 1
        self.biases['B_r'] = tf.get_variable("B_r", shape=[
            hidden_state_dim, 1], initializer=tf.contrib.layers.xavier_initializer())  # H x 1
        self.biases['B_h'] = tf.get_variable("B_h", shape=[
            hidden_state_dim, 1], initializer=tf.contrib.layers.xavier_initializer())  # H x 1
        self.biases['message'] = tf.get_variable("bias_message", shape=[
            hidden_state_dim, 1], initializer=tf.contrib.layers.xavier_initializer())  # H x 1

        self.weights['W_h_v'] = tf.get_variable("W_h_v", shape=[
                                                num_verbs, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())
        self.weights['W_h_n'] = tf.get_variable("W_h_n", shape=[
                                                num_nouns, hidden_state_dim], initializer=tf.contrib.layers.xavier_initializer())

        self.biases['B_h_v'] = tf.get_variable(
            "B_h_v", shape=[num_verbs, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.biases['B_h_n'] = tf.get_variable(
            "B_h_n", shape=[num_nouns, 1], initializer=tf.contrib.layers.xavier_initializer())
