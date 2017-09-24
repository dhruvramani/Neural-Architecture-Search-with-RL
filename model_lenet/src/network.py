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
        self.Wconv1, self.bconv1, self.Wconv2, self.bconv2, self.Wconv3, self.bconv3 =  None, None, None, None, None, None
        self.Wf1, self.bf1, self.Wf2, self.bf2, self.Wf3, self.bf3, self.Wf4, self.bf4 = None, None, None, None, None, None, None, None
    
    def weight_variable(self, shape, name):
        return tf.Variable(tf.random_normal(shape=shape), name=name)

    def bias_variable(self, shape, name):
        return tf.Variable(tf.random_normal(shape=shape), name=name)

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
        out = [utils.max(output[i]) for i in range(self.n_steps)]
        return out, output[-1]

    def gen_hyperparams(self, output):
        filter_dims = tf.constant([3, 5, 7, 9], dtype=tf.int32)
        n_filters = tf.constant([24, 36, 48, 64], dtype=tf.int32)
        strides = tf.constant([1, 2, 3, 4], dtype=tf.int32)
        hyperparams = [1 for _ in range(self.n_steps)]
        # Change the following based on number of hyperparameters to be predicted
        # Removing strides for now
        hyperparams[0], hyperparams[1] = filter_dims[output[0]], filter_dims[output[1]]
        hyperparams[2] = n_filters[output[2]]  # Layer 1
        hyperparams[3], hyperparams[4] = filter_dims[output[3]], filter_dims[output[5]]
        hyperparams[5] = n_filters[output[5]]  # Layer 2
        hyperparams[6], hyperparams[7] = filter_dims[output[6]], filter_dims[output[7]]
        hyperparams[8] = n_filters[output[8]] # Layer 3
        hyperparams[9] = n_filters[output[9]] # FNN Layer

        return hyperparams

    def construct_model(self, data, hyperparams, keep_prob, inside=1):
        data = tf.expand_dims(data, -1)
        data = tf.expand_dims(data, -1)
        hyperparams = {"filter_row_1": hyperparams[0], "filter_column_1": hyperparams[1], "n_filter_1": hyperparams[2], "filter_row_2": hyperparams[3], "filter_column_2": hyperparams[4], "n_filter_2": hyperparams[5], "filter_row_3": hyperparams[6], "filter_column_3": hyperparams[7], "n_filter_3": hyperparams[8], "n_autoneurons": hyperparams[9]}
        images = tf.reshape(data, [self.config.batch_size, 32, 32, 3])
        if(inside == 0):
            self.Wconv1 = self.weight_variable(shape=[hyperparams["filter_row_1"], hyperparams["filter_column_1"], 3, hyperparams["n_filter_1"]], name="kernel_1")
            self.bconv1 = self.bias_variable(shape=[hyperparams["n_filter_1"]], name="b_conv_1")
            self.Wconv2 = self.weight_variable(shape=[hyperparams["filter_row_2"], hyperparams["filter_column_2"], hyperparams["n_filter_1"], hyperparams["n_filter_2"]], name="kernel_2")
            self.bconv2 = self.bias_variable(shape=[hyperparams["n_filter_2"]], name="b_conv_2")
            self.Wconv3 = self.weight_variable(shape=[hyperparams["filter_row_3"], hyperparams["filter_column_3"], hyperparams["n_filter_2"], hyperparams["n_filter_3"]], name="kernel_3")
            self.bconv3 = self.bias_variable(shape=[hyperparams["n_filter_3"]], name="b_conv_3")
            self.bf1 = self.bias_variable(shape=[384], name="b_fc1")
            self.Wf2 = self.weight_variable(shape=[384, 192], name="w_fc2")
            self.bf2 = self.bias_variable(shape=[192], name="b_fc2")
            self.Wf3 = self.weight_variable(shape=[192, hyperparams["n_autoneurons"]], name="w_fc3")
            self.bf3 = self.bias_variable(shape=[hyperparams["n_autoneurons"]], name="b_fc3")
            self.Wf4 = self.weight_variable(shape=[hyperparams["n_autoneurons"], self.config.num_classes], name="w_fc4")
            self.bf4 = self.bias_variable(shape=[self.config.num_classes], name="b_fc4")

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
        #mult = 1
        #for i in shape[1:]:
        #    mult *= i
        reshaped = tf.reshape(pool3, [shape[0], -1])
        dim = reshaped.get_shape()[1].value
        if(inside == 0):
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
