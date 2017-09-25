import os
import sys
import utils
import numpy as np
import tensorflow as tf

class ChildNetwork(object):
    def __init__(self, config, hyperparams):
        self.config = config
        self.hyperparams = {"filter_row_1": hyperparams[0], "filter_column_1": hyperparams[1], "n_filter_1": hyperparams[2], "filter_row_2": hyperparams[3], "filter_column_2": hyperparams[4], "n_filter_2": hyperparams[5], "filter_row_3": hyperparams[6], "filter_column_3": hyperparams[7], "n_filter_3": hyperparams[8], "n_autoneurons": hyperparams[9]}
        self.Wconv1, self.bconv1, self.Wconv2, self.bconv2, self.Wconv3, self.bconv3 = self.init_conv_vars()
        self.Wf1, self.bf1, self.Wf2, self.bf2, self.Wf3, self.bf3, self.Wf4, self.bf4 = self.init_fc_vars()

    def weight_variable(self, shape, name):
        return tf.Variable(tf.random_normal(shape=shape), name=name)

    def bias_variable(self, shape, name):
        return tf.Variable(tf.random_normal(shape=shape), name=name)

    def init_conv_vars(self):
        Wconv1 = self.weight_variable(shape=[self.hyperparams["filter_row_1"], self.hyperparams["filter_column_1"], 3, self.hyperparams["n_filter_1"]], name="kernel_1")
        bconv1 = self.bias_variable(shape=[self.hyperparams["n_filter_1"]], name="b_conv_1")
        Wconv2 = self.weight_variable(shape=[self.hyperparams["filter_row_2"], self.hyperparams["filter_column_2"], self.hyperparams["n_filter_1"], self.hyperparams["n_filter_2"]], name="kernel_2")
        bconv2 = self.bias_variable(shape=[self.hyperparams["n_filter_2"]], name="b_conv_2")
        Wconv3 = self.weight_variable(shape=[self.hyperparams["filter_row_3"], self.hyperparams["filter_column_3"], self.hyperparams["n_filter_2"], self.hyperparams["n_filter_3"]], name="kernel_3")
        bconv3 = self.bias_variable(shape=[self.hyperparams["n_filter_3"]], name="b_conv_3")
        return Wconv1, bconv1, Wconv2, bconv2, Wconv3, bconv3

    def init_fc_vars(self):
        Wf1 = self.weight_variable(shape=[16, 384], name="w_fc1")
        bf1 = self.bias_variable(shape=[384], name="b_fc1")
        Wf2 = self.weight_variable(shape=[384, 192], name="w_fc2")
        bf2 = self.bias_variable(shape=[192], name="b_fc2")
        Wf3 = self.weight_variable(shape=[192, self.hyperparams["n_autoneurons"]], name="w_fc3")
        bf3 = self.bias_variable(shape=[self.hyperparams["n_autoneurons"]], name="b_fc3")
        Wf4 = self.weight_variable(shape=[self.hyperparams["n_autoneurons"], self.config.num_classes], name="w_fc4")
        bf4 = self.bias_variable(shape=[self.config.num_classes], name="b_fc4")
        return Wf1, bf1, Wf2, bf2, Wf3, bf3, Wf4, bf4

    def run_model(self, data, keep_prob):
        data = tf.expand_dims(data, -1)
        data = tf.expand_dims(data, -1)
        images = tf.reshape(data, [self.config.batch_size, 32, 32, 3])

        conv1 = tf.nn.conv2d(images, self.Wconv1, strides=[1, 1, 1, 1], padding="SAME")
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.bconv1))
        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding = 'SAME')        
        conv2 = tf.nn.conv2d(pool1, self.Wconv2, strides=[1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.bconv2))
        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding = 'SAME')
        conv3 = tf.nn.conv2d(pool2, self.Wconv3, strides=[1, 1, 1, 1], padding="SAME")
        conv3 = tf.nn.relu(tf.nn.bias_add(conv3, self.bconv3))
        pool3 = tf.nn.max_pool(conv3, [1, 3, 3, 1], [1, 2, 2, 1], padding = 'SAME')

        shape = pool3.get_shape().as_list()
        reshaped = tf.reshape(pool3, [shape[0], -1])
        dim = reshaped.get_shape()[1].value
        self.Wf1 = self.weight_variable(shape=[dim, 384], name="w_fc1")
        f1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(reshaped, self.Wf1) + self.bf1), keep_prob)
        f2 = tf.nn.dropout(utils.leaky_relu(tf.matmul(f1, self.Wf2) + self.bf2), keep_prob)
        fc = tf.nn.dropout(utils.leaky_relu(tf.matmul(f2, self.Wf3) + self.bf3), keep_prob)
        output = tf.matmul(fc, self.Wf4)+ self.bf4
        return output

    def model_loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def train_model(self, loss):
        optimizer = self.config.solver.optimizer
        var_list = [self.Wconv1, self.bconv1, self.Wconv2, self.bconv2, self.Wconv3, self.bconv3, self.Wf1, self.bf1, self.Wf2, self.bf2, self.Wf3, self.bf3, self.Wf4, self.bf4]
        return optimizer.minimize(loss, var_list=var_list)

    def accuracy(self, logits, labels):
        return 100.0 * tf.reduce_mean(tf.cast(tf.equal(utils.max(logits), utils.max(labels)), tf.float32))

