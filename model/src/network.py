import os
import utils
import numpy as np
import tensorflow as tf

class Network(object):
    def __init__(self, config):
        self.config = config
        self.n_steps = self.config.hyperparams
        self.n_input, self.n_hidden =  4, 20
        self.lstm = tf.contrb.rnn.BasicLSTMCell(self.n_hidden)
        self.Wc, self.bc = self.init_controller_vars()
        self.Wconv, self.bconv, self.Wf1, self.bf1, self.Wf2, self.bf2 = None, None, None, None, None, None
    
    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def init_controller_vars(self):
        Wc = self.weight_variable(shape=[self.n_hidden, self.n_input], name="w_controller")
        bc = self.bias_variable(shape=[self.n_input], name="b_controller")
        return Wc, bc

    def neural_search(self):
        state = tf.Variable(tf.zeros(shape=[4]))
        inp = tf.Variable(tf.random_normal(shape=[4]))
        output = list()
        for _ in n_steps:
            inp, state = self.lstm(inp, state)
            inp = tf.nn.softmax(tf.matmul(inp, self.Wc) + self.bc)
            output.append(inp)
        out = [utils.max(output[0]), utils.max(output[1]), utils.max(output[2]), utils.max(output[3]), utils.max(output[4]), utils.max(output[5])]
        return out, output[-1]

    def gen_hyperparams(self, output):
        filter_dims = tf.constant([1, 3, 5, 7], dtype=tf.int32)
        n_filters = tf.constant([24, 36, 48, 64], dtype=tf.int32)
        strides = tf.constant([1, 2, 3, 4], dtype=tf.int32)
        hyperparams = {"filter_row": filter_dims[output[0]], "filter_column": filter_dims[output[1]], "stride_height": strides[output[2]], "stride_width": strides[output[3]], "n_filter": n_filters[output[4]], "n_autoneurons": n_filters[output[5]]}
        return hyperparams

    def construct_model(self, data, hyperparams, keep_prob):
        # Current DATA : batch_size, 3072
        data = tf.expand_dims(data, -1)
        data = tf.expand_dims(data, -1)
        images = tf.reshape(data, [self.config.batch_size, 32, 32, 3])
        self.Wconv = weight_variable(shape=[hyperparams["filter_row"], hyperparams["filter_column"], 3, hyperparams["n_filter"]], name="kernel")
        self.bconv = tf.Variable(tf.random_normal(shape=hyperparams["n_filter"]), name="b_conv")
        self.Wf1 = weight_variable(shape=[32*32*3, hyperparams["n_autoneurons"]], "w_fc1")
        self.bf1 = bias_variable(shape=[hyperparams["n_autoneurons"]], "b_fc1")
        self.Wf2 = weight_variable(shape=[hyperparams["n_autoneurons"], 10], "w_fc2")
        self.bf1 = bias_variable(shape=[10], "b_fc2")
        convlove = tf.nn.conv2d(images, self.Wconv, strides=[1, hyperparams["stride_height"], 1], padding="SAME")
        convlove = tf.nn.bias_add(convlove, self.bconv)
        reshaped = tf.reshape(convlove, [self.config.batch_size, 32*32*3])
        fc = tf.nn.dropout(utils.leaky_relu(tf.matmul(reshaped, self.Wf1) + self.Wf1), keep_prob)
        output = tf.matmul(fc, self.Wf2)+ self.bf2
        return output

    def model_loss(self, logits, labels):
        return -1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def train_model(self, loss):
        optimizer = self.config.solver.optimizer
        var_list = [self.Wconv, self.bconv, self.Wf1, self.bf1, self.Wf2, self.bf2]
        return optimizer(self.config.solver.learning_rate).minimize(loss, var_list=var_list)

    def accuracy(self, logits, labels):
        return tf.reduce_mean(tf.cast(tf.equal(utils.max(logits), utils.max(labels)), tf.float32))

    def train_controller(self, reinforce_loss, val_accuracy)
        optimizer = self.config.solver.optimizer
        var_list = [self.Wc, self.bc]
        gradients = optimizer.compute_gradients(reinforce_loss, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (grad * val_accuracy, var)
        return optimizer.apply_gradients(gradients)

    def REINFORCE(self, prob):
        loss = tf.reduce_mean(tf.log(prob)) # Might have to take the negative
        return loss
