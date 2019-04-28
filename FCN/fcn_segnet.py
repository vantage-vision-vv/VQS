import tensorflow as tf
import numpy as np
import math


class segnet(object):
    def __init__(self):
        self.num_classes = 2
        self.image_height = 224
        self.image_weight = 224
        self.image_channel = 3  # check once
        self.bayes = True
        self.learning_rate = 1e-3  # feed

        self.dropout_bool = True
        self.keep_rate = 0.6
        self.is_training = True
        self.use_vgg = False
        self.vgg_param_dict = None
        self.batch_size = 1

    def forward(self):
        self.inputs_pl = tf.placeholder(
            tf.float32, [None, self.image_height, self.image_weight, self.image_channel], name="input")
        self.labels_pl = tf.placeholder(
            tf.int64, [None, self.image_height, self.image_weight, 1])
        self.att_map_pl = tf.placeholder(tf.int64, [None, 7, 7, 1])

        self.norm1 = tf.nn.lrn(self.inputs_pl, depth_radius=5,
                               bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

        # First box of convolution layer(1)
        self.conv1_1 = self.conv_layer(self.norm1, "conv1_1", [
            3, 3, 3, 64], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2", [
            3, 3, 64, 64], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.pool1, self.pool1_index, self.shape_1 = self.max_pool(
            self.conv1_2, 'pool1')

        # Second box of convolution layer(4)
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", [
            3, 3, 64, 128], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", [
            3, 3, 128, 128], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.pool2, self.pool2_index, self.shape_2 = self.max_pool(
            self.conv2_2, 'pool2')

        # Third box of convolution layer(7)
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1", [
            3, 3, 128, 256], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2", [
            3, 3, 256, 256], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3", [
            3, 3, 256, 256], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.pool3, self.pool3_index, self.shape_3 = self.max_pool(
            self.conv3_3, 'pool3')

        # Fourth box of convolution layer(10)
        if self.bayes:
            self.dropout1 = tf.layers.dropout(self.pool3, rate=(
                1 - self.keep_rate), training=self.dropout_bool, name="dropout1")
            self.conv4_1 = self.conv_layer(self.dropout1, "conv4_1", [
                3, 3, 256, 512], self.is_training, self.use_vgg, self.vgg_param_dict)
        else:
            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1", [
                3, 3, 256, 512], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2", [
            3, 3, 512, 512], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3", [
            3, 3, 512, 512], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.pool4, self.pool4_index, self.shape_4 = self.max_pool(
            self.conv4_3, 'pool4')

        # Fifth box of convolution layers(13)
        if self.bayes:
            self.dropout2 = tf.layers.dropout(self.pool4, rate=(1 - self.keep_rate),
                                              training=self.dropout_bool, name="dropout2")
            self.conv5_1 = self.conv_layer(self.dropout2, "conv5_1", [
                3, 3, 512, 512], self.is_training, self.use_vgg, self.vgg_param_dict)
        else:
            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1", [
                3, 3, 512, 512], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2", [
            3, 3, 512, 512], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3", [
            3, 3, 512, 512], self.is_training, self.use_vgg, self.vgg_param_dict)
        self.pool5, self.pool5_index, self.shape_5 = self.max_pool(
            self.conv5_3, 'pool5')

        # ---------------------So Now the encoder process has been Finished--------------------------------------#
        self.metadata = tf.math.multiply(self.pool5, self.att_map_pl)
        # ------------------Then Let's start Decoder Process-----------------------------------------------------#

        # First box of deconvolution layers(3)
        if self.bayes:
            self.dropout3 = tf.layers.dropout(self.metadata, rate=(
                1 - self.keep_rate), training=self.dropout_bool, name="dropout3")
            self.deconv5_1 = self.up_sampling(
                self.dropout3, self.pool5_index, self.shape_5, self.batch_size, name="unpool_5")
        else:
            self.deconv5_1 = self.up_sampling(
                self.metadata, self.pool5_index, self.shape_5, self.batch_size, name="unpool_5")
        self.deconv5_2 = self.conv_layer(self.deconv5_1, "deconv5_2", [
            3, 3, 512, 512], self.is_training)
        self.deconv5_3 = self.conv_layer(self.deconv5_2, "deconv5_3", [
            3, 3, 512, 512], self.is_training)
        self.deconv5_4 = self.conv_layer(self.deconv5_3, "deconv5_4", [
            3, 3, 512, 512], self.is_training)
        # Second box of deconvolution layers(6)
        if self.bayes:
            self.dropout4 = tf.layers.dropout(self.deconv5_4, rate=(
                1 - self.keep_rate), training=self.dropout_bool, name="dropout4")
            self.deconv4_1 = self.up_sampling(
                self.dropout4, self.pool4_index, self.shape_4, self.batch_size, name="unpool_4")
        else:
            self.deconv4_1 = self.up_sampling(
                self.deconv5_4, self.pool4_index, self.shape_4, self.batch_size, name="unpool_4")
        self.deconv4_2 = self.conv_layer(self.deconv4_1, "deconv4_2", [
            3, 3, 512, 512], self.is_training)
        self.deconv4_3 = self.conv_layer(self.deconv4_2, "deconv4_3", [
            3, 3, 512, 512], self.is_training)
        self.deconv4_4 = self.conv_layer(self.deconv4_3, "deconv4_4", [
            3, 3, 512, 256], self.is_training)
        # Third box of deconvolution layers(9)
        if self.bayes:
            self.dropout5 = tf.layers.dropout(self.deconv4_4, rate=(
                1 - self.keep_rate), training=self.dropout_bool, name="dropout5")
            self.deconv3_1 = self.up_sampling(
                self.dropout5, self.pool3_index, self.shape_3, self.batch_size, name="unpool_3")
        else:
            self.deconv3_1 = self.up_sampling(
                self.deconv4_4, self.pool3_index, self.shape_3, self.batch_size, name="unpool_3")
        self.deconv3_2 = self.conv_layer(self.deconv3_1, "deconv3_2", [
            3, 3, 256, 256], self.is_training)
        self.deconv3_3 = self.conv_layer(self.deconv3_2, "deconv3_3", [
            3, 3, 256, 256], self.is_training)
        self.deconv3_4 = self.conv_layer(self.deconv3_3, "deconv3_4", [
            3, 3, 256, 128], self.is_training)
        # Fourth box of deconvolution layers(11)
        if self.bayes:
            self.dropout6 = tf.layers.dropout(self.deconv3_4, rate=(
                1 - self.keep_rate), training=self.dropout_bool, name="dropout6")
            self.deconv2_1 = self.up_sampling(
                self.dropout6, self.pool2_index, self.shape_2, self.batch_size, name="unpool_2")
        else:
            self.deconv2_1 = self.up_sampling(
                self.deconv3_4, self.pool2_index, self.shape_2, self.batch_size, name="unpool_2")
        self.deconv2_2 = self.conv_layer(self.deconv2_1, "deconv2_2", [
            3, 3, 128, 128], self.is_training)
        self.deconv2_3 = self.conv_layer(self.deconv2_2, "deconv2_3", [
            3, 3, 128, 64], self.is_training)
        # Fifth box of deconvolution layers(13)
        self.deconv1_1 = self.up_sampling(
            self.deconv2_3, self.pool1_index, self.shape_1, self.batch_size, name="unpool_1")
        self.deconv1_2 = self.conv_layer(self.deconv1_1, "deconv1_2", [
            3, 3, 64, 64], self.is_training)
        self.deconv1_3 = self.conv_layer(self.deconv1_2, "deconv1_3", [
            3, 3, 64, 64], self.is_training)

        with tf.variable_scope('conv_classifier') as scope:
            self.kernel = self.variable_with_weight_decay('weights', initializer=self.initialization(
                1, 64), shape=[1, 1, 64, self.num_classes], wd=False)
            self.conv = tf.nn.conv2d(self.deconv1_3, self.kernel, [
                                     1, 1, 1, 1], padding='SAME')
            self.biases = self.variable_with_weight_decay('biases', tf.constant_initializer(0.0),
                                                          shape=[self.num_classes], wd=False)
            #self.logits = tf.nn.bias_add(self.conv, self.biases, name=scope.name)
            self.logits = tf.nn.bias_add(self.conv, self.biases, name="output")

        return self.logits

    def max_pool(self, inputs, name):
        with tf.variable_scope(name) as scope:
            value, index = tf.nn.max_pool_with_argmax(tf.to_double(inputs), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                      padding='SAME', name=scope.name)
        return tf.to_float(value), index, inputs.get_shape().as_list()

    def conv_layer(self, bottom, name, shape, is_training, use_vgg=False, vgg_param_dict=None):
        with tf.variable_scope(name) as scope:
            filt = self.variable_with_weight_decay('weights', initializer=self.initialization(
                shape[0], shape[2]), shape=shape, wd=False)
            tf.summary.histogram(scope.name + "weight", filt)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.variable_with_weight_decay(
                'biases', initializer=tf.constant_initializer(0.0), shape=shape[3], wd=False)
            tf.summary.histogram(scope.name + "bias", conv_biases)
            bias = tf.nn.bias_add(conv, conv_biases)
            conv_out = tf.nn.relu(self.batch_norm(bias, is_training, scope))
        return conv_out

    def up_sampling(self, pool, ind, output_shape, batch_size, name=None):
        """
        Unpooling layer after max_pool_with_argmax.
        Args:
            pool:   max pooled output tensor
            ind:      argmax indices
            ksize:     ksize is the same as for the pool
        Return:
            unpool:    unpooling tensor
            :param batch_size:
        """
        with tf.variable_scope(name):
            pool_ = tf.reshape(pool, [-1])
            batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), [
                tf.shape(pool)[0], 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, [-1, 1])
            ind_ = tf.reshape(ind, [-1, 1])
            ind_ = tf.concat([b, ind_], 1)
            ret = tf.scatter_nd(ind_, pool_, shape=[
                                batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
            # the reason that we use tf.scatter_nd: if we use tf.sparse_tensor_to_dense, then the gradient is None, which will cut off the network.
            # But if we use tf.scatter_nd, the gradients for all the trainable variables will be tensors, instead of None.
            # The usage for tf.scatter_nd is that: create a new tensor by applying sparse UPDATES(which is the pooling value) to individual values of slices within a
            # zero tensor of given shape (FLAT_OUTPUT_SHAPE) according to the indices (ind_). If we ues the orignal code, the only thing we need to change is: changeing
            # from tf.sparse_tensor_to_dense(sparse_tensor) to tf.sparse_add(tf.zeros((output_sahpe)),sparse_tensor) which will give us the gradients!!!
            ret = tf.reshape(
                ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
            return ret

    def initialization(self, k, c):
        """
        Here the reference paper is https:arxiv.org/pdf/1502.01852
        k is the filter size
        c is the number of input channels in the filter tensor
        we assume for all the layers, the Kernel Matrix follows a gaussian distribution N(0, \sqrt(2/nl)), where nl is 
        the total number of units in the input, k^2c, k is the spartial filter size and c is the number of input channels. 
        Output:
        The initialized weight
        """
        std = math.sqrt(2. / (k ** 2 * c))
        return tf.truncated_normal_initializer(stddev=std)

    def variable_with_weight_decay(self, name, initializer, shape, wd):
        """
        Help to create an initialized variable with weight decay
        The variable is initialized with a truncated normal distribution, only the value for standard deviation is determined
        by the specific function _initialization
        Inputs: wd is utilized to notify if weight decay is utilized
        Return: variable tensor
        """
        var = tf.get_variable(name, shape, initializer=initializer)
        if wd is True:
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        # tf.nn.l2_loss is utilized to compute half L2 norm of a tensor without the sqrt output = sum(t**2)/2
        return var

    def batch_norm(self, bias_input, is_training, scope):
        with tf.variable_scope(scope.name) as scope:
            return tf.cond(is_training, lambda: tf.contrib.layers.batch_norm(bias_input, is_training=True, center=False, scope=scope), lambda: tf.contrib.layers.batch_norm(bias_input, is_training=False, center=False, reuse=True, scope=scope))
