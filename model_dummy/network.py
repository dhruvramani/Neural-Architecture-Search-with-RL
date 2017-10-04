import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from keras import backend as K
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Dropout

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True)

#def train_rl(loss, father_lr, val_accuracy):

def father_network():
    father_lr = 0.01
    inputdim, timesteps, classes =  4, 2, 4
    inp = tf.constant(np.random.random((1, timesteps, inputdim)).astype(np.float32))
    X = LSTM(classes, return_sequences=True)(inp)

    hyperparams = Dense(classes, activation='softmax')(X) # [1, 2, 4]
    loss = tf.reduce_mean(tf.log(hyperparams))
    val_accuracy = tf.placeholder_with_default(10.0, shape=())
    
    optimizer = tf.train.GradientDescentOptimizer(father_lr)
    gradients = optimizer.compute_gradients(loss=loss)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (grad * val_accuracy, var)
    train = optimizer.apply_gradients(gradients)
    
    hidden_layers = [100, 300, 600, 900]
    learning_rates = [0.01, 0.1, 1.0, 3.0]
    
    with tf.Session() as sess:
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())
        hyp = np.ones((1, timesteps, classes))
        print("\n\n")
        for i in range(10000):
            print("RL {}".format(i))
            hyp = [np.argmax(hyp[0, i, :]) for i in range(timesteps)]
            no_hidden, lr = hidden_layers[hyp[0]], learning_rates[hyp[1]]
            val_acc = train_network(no_hidden, lr)
            output = "\nController Loss : {}\n".format(sess.run(loss))
            output += "Accuracy : {}, Learning Rate : {}, Hidden Number : {}\n".format(val_acc, lr, no_hidden)
            with open("accuracy.log", "a+") as f:
                f.write(output)
            print(output)
            _ = sess.run(train, feed_dict = {val_accuracy : val_acc})
            hyp = sess.run(hyperparams)

def train_network(no_hidden=600, learning_rate=3.0):
    no_input, no_output = 784, 10
    val_accuracy = 0

    x = tf.placeholder(tf.float32 ,shape = [None, no_input])
    y = tf.placeholder(tf.float32 , shape = [None, no_output])

    W1 = tf.Variable(tf.random_normal(shape = [no_input, no_hidden])) # Used Theta1'
    W2 = tf.Variable(tf.random_normal(shape = [no_hidden ,no_output]))
    b1 = tf.Variable(tf.random_normal(shape = [1, no_hidden]))
    b2 = tf.Variable(tf.random_normal(shape = [1, no_output]))
        
    h = tf.matmul(tf.nn.sigmoid(tf.matmul(x, W1) + b1), W2) + b2

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = h,labels = y))
    gradient = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    equal = tf.equal(tf.argmax(y, 1) , tf.argmax(h, 1))
    accuracy = tf.reduce_mean(tf.cast(equal , tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(700):  
            batch_X , batch_Y = mnist.train.next_batch(100)
            loss, _ = sess.run([cost, gradient], feed_dict = {x : batch_X , y : batch_Y})
            print("Child Loss : {}".format(loss), end="\r")
        val_accuracy = sess.run(accuracy, feed_dict = {x : mnist.validation.images, y : mnist.validation.labels})
    return val_accuracy

if __name__ == '__main__':
    father_network()