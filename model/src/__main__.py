import os
import sys
import utils
import numpy as np
import tensorflow as tf
from parser import Parser
from config import Config
from dataset import DataSet
from network import Network

class Model(object):
    def __init__(self, config):
        self.config = config
        self.data = DataSet(config)
        self.add_placeholders()
        self.summarizer = tf.summary
        self.net = Network(config)
        self.epoch_count, self.second_epoch_count = 0, 0
        self.outputs, self.prob = self.net.neural_search()
        self.hyperparams = self.net.gen_hyperparams(self.outputs)
        self.y_pred = self.net.construct_model(self.X, self.hyperparams, self.keep_prob)
        self.cross_loss = self.net.model_loss(self.y_pred, self.Y)
        self.tr_model_step = self.net.train_model(self.cross_loss)
        self.accuracy = self.net.accuracy(self.y_pred, self.Y)
        self.reinforce_loss = self.net.REINFORCE(self.val_accuracy, self.prob)
        self.tr_cont_step = self.net.train_controller(self.reinforce_loss)

    def add_placeholders(self):
        self.X = tf.placeholder(tf.float32, shape=[None, 3072])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])
        self.val_accuracy = tf.placeholder(tf.float32)
        self.dropout = tf.placeholder(tf.float32)

    def run_model_epoch(self, sess, data, summarizer, epoch):
        X, Y, i, err= None, None, 0, list()
        merged_summary = self.summarizer.merge_all()
        for X, Y, tot in self.data.next_batch(data):
            feed_dict = {self.X : X, self.Y : Y, self.keep_prob : self.config.solver.dropout}
            summ, _, loss = sess.run([merged_summary, self.tr_model_step, self.cross_loss], feed_dict=feed_dict)
            output = "Epoch ({}.{}) Batch({}) : Loss = {}".format(self.epoch_count, self.second_epoch_count, i , loss)
            with open("../stdout/train.log", "a+") as log:
                log.write(output + "\n")
            print("   {}".format(output), end='\r')
            step = int(epoch*tot + i)
            err.append(loss) 
            summarizer.add_summary(summ, step)
            i += 1
        err = np.asarray(err)
        return np.mean(err), step

    def run_model_eval(self, sess, data="validation", summary_writer=None, step=0):
        y, y_pred, loss_, loss, i, acc, accuracy = list(), list(), 0.0, 0.0, 0, 0.0, 0.0
        merged_summary = self.summarizer.merge_all()
        for X, Y, tot in self.data.next_batch(data):
            feed_dict = {self.X: X, self.Y: Y, self.keep_prob: 1.0}
                summ, loss_, acc =  sess.run([merged_summary, self.cross_loss, self.accuracy], feed_dict=feed_dict)
                summary_writer.add_summary(summ, step)
            loss += loss_
            accuracy += acc
            i += 1
        return loss / i, accuracy / i

    def add_summaries(self, sess):
        if self.config.load or self.config.debug:
            path_ = "../results/tensorboard"
        else :
            path_ = "../bin/results/tensorboard"
        summary_writer_train = tf.summary.FileWriter(path_ + "/train", sess.graph)
        summary_writer_val = tf.summary.FileWriter(path_ + "/val", sess.graph)
        summary_writer_test = tf.summary.FileWriter(path_+ "/test", sess.graph)
        summary_writers = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers

    def fit(self, sess, summarizer):
        max_epochs = self.config.max_epochs
        self.epoch_count, self.second_epoch_count = 0, 0
        losses, learning_rate, val_accuracy = list(), self.config.solver.learning_rate, 0.0
        while self.epoch_count < max_epochs:
            while self.second_epoch_count < max_epochs :
                average_loss, tr_step = self.run_model_epoch(sess, "train", summarizer['train'], self.epoch_count)
                if not self.config.debug:
                    val_loss, val_accuracy = self.run_model_eval(sess, "validation", summarizer['val'], tr_step)
                    output =  "=> Training : Loss = {:.3f} | Validation : Loss = {:.3f}".format(average_loss, val_loss)
                    with open("../stdout/validation.log", "a+") as f:
                        f.write(output_)
                    print(output)
                self.second_epoch_count += 1
            _ = sess.run(self.tr_cont_step, feed_dict={self.val_accuracy : val_accuracy})
            test_loss, test_accuracy = self.run_model_eval(sess, "test", summarizer['test'], tr_step)
            self.epoch_count += 1
        returnDict = {"test_loss" : test_loss, "test_accuracy" : test_accuracy}
        return returnDict

def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    with tf.variable_scope('Model', reuse=None) as scope:
        model = Model(config)

    tf_config = tf.ConfigProto(allow_soft_placement=True)#, device_count = {'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain or config.load == True:
        print("=> Loading model from checkpoint")
        load_ckpt_dir = config.ckptdir_path
    else:
        print("=> No model loaded from checkpoint")
        load_ckpt_dir = ''
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tf_config)
    return model, sess

def train_model(config):
    print("\033[92m=>\033[0m Training Model")
    model, sess = init_model(config)
    with sess:
        summary_writers = model.add_summaries(sess)
        loss_dict = model.fit(sess, summary_writers)
        return loss_dict

def main():
    args = Parser().get_parser().parse_args()
    config = Config(args)
    loss_dict = train_model(config)
    output = "=> Test Loss : {}, Test Accuracy : {}".format(loss_dict["test_loss"], loss_dict["test_accuracy"])
    with open("../stdout/test_log.log", "a+") as f:
        f.write(output)
    print("\033[1m\033[92m{}\033[0m\033[0m".format(output))

if __name__ == '__main__' :
    np.random.seed(1234)
    main()