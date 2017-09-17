import os
import sys
import utils
import numpy as np
import tensorflow as tf

class Network(object):
    def __init__(self, config):
        self.config = config
        self.n_steps = self.config.hyperparams
        self.n_input, self.n_hidden =  4, 2
        self.state = tf.Variable(tf.zeros(shape=[1, 4]))
        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=False)
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
        inp = tf.constant(np.zeros((1, 4), dtype="float32"))
        output = list()
        for _ in range(self.n_steps):
            inp, self.state = self.lstm(inp, self.state)
            inp = tf.nn.softmax(tf.matmul(inp, self.Wc) + self.bc)
            output.append(inp[0, :])
        out = [utils.max(output[i]) for i in range(6)]
        return out, output[-1]

    def gen_hyperparams(self, output):
        filter_dims = tf.constant([1, 3, 5, 7], dtype=tf.int32)
        n_filters = tf.constant([24, 36, 48, 64], dtype=tf.int32)
        strides = tf.constant([1, 2, 3, 4], dtype=tf.int32)
        hyperparams = [filter_dims[output[i]] for i in range(6)]
        return hyperparams

    def construct_model(self, data, hyperparams, keep_prob):
        data = tf.expand_dims(data, -1)
        data = tf.expand_dims(data, -1)
        hyperparams = {"filter_row": hyperparams[0], "filter_column": hyperparams[1], "stride_height": hyperparams[2], "stride_width": hyperparams[3], "n_filter": hyperparams[4], "n_autoneurons": hyperparams[5]}
        images = tf.reshape(data, [self.config.batch_size, 32, 32, 3])
        self.Wconv = self.weight_variable(shape=[hyperparams["filter_row"], hyperparams["filter_column"], 3, hyperparams["n_filter"]], name="kernel")
        self.bconv = self.bias_variable(shape=[hyperparams["n_filter"]], name="b_conv")
        self.bf1 = self.bias_variable(shape=[hyperparams["n_autoneurons"]], name="b_fc1")
        self.Wf2 = self.weight_variable(shape=[hyperparams["n_autoneurons"], 10], name="w_fc2")
        self.bf2 = self.bias_variable(shape=[10], name="b_fc2")
        convlove = tf.nn.conv2d(images, self.Wconv, strides=[1, hyperparams["stride_height"], hyperparams["stride_width"], 1], padding="SAME")
        convlove = tf.nn.bias_add(convlove, self.bconv)
        shape = convlove.get_shape().as_list()
        mult = shape[1] * shape[2] * shape[3] 
        self.Wf1 = self.weight_variable(shape=[mult, hyperparams["n_autoneurons"]], name="w_fc1")
        reshaped = tf.reshape(convlove, [shape[0], mult])
        fc = tf.nn.dropout(utils.leaky_relu(tf.matmul(reshaped, self.Wf1) + self.bf1), keep_prob)
        output = tf.matmul(fc, self.Wf2)+ self.bf2
        return output

    def model_loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def train_model(self, loss):
        optimizer = self.config.solver.optimizer
        var_list = [self.Wconv, self.bconv, self.Wf1, self.bf1, self.Wf2, self.bf2]
        return optimizer.minimize(loss, var_list=var_list)

    def accuracy(self, logits, labels):
        return tf.reduce_mean(tf.cast(tf.equal(utils.max(logits), utils.max(labels)), tf.float32))

    def train_controller(self, reinforce_loss, val_accuracy):
        optimizer = self.config.solver.optimizer
        var_list = [self.Wc, self.bc]
        gradients = optimizer.compute_gradients(loss=reinforce_loss, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (grad * val_accuracy, var)
        return optimizer.apply_gradients(gradients)

    def REINFORCE(self, prob):
        loss = tf.reduce_mean(tf.log(prob)) # Might have to take the negative
        return loss
