import os
import numpy as np
import tensorflow as tf
from keras.layers import LSTM
from keras import backend as K
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tensorflow.examples.tutorials.mnist import input_data

inputdim, timesteps, classes =  4, 2, 4
mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True)

def father_network():
    father_lr = 7.0
    inp = tf.placeholder(tf.float32, shape=[1, timesteps, inputdim])
    X = LSTM(classes, return_sequences=True)(inp)

    hyperparams = Dense(classes, activation='softmax')(X) # [1, timesteps, classes]
    loss = - tf.reduce_mean(tf.log(1e-10 + hyperparams))
    val_accuracy = tf.placeholder_with_default(10.0, shape=())
    
    optimizer = tf.train.RMSPropOptimizer(father_lr)
    gradients = optimizer.compute_gradients(loss=loss)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (grad * val_accuracy, var)
    train = optimizer.apply_gradients(gradients)
    
    hidden_layers = [100, 300, 600, 900]
    learning_rates = [0.01, 0.1, 1.0, 3.0]
    
    hyp = np.random.random((1, timesteps, inputdim)).astype(np.float32)
    with tf.Session() as sess:
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())
        print("\n\n")
        for i in range(10000):
            print("Controller Epoch # {}".format(i))
            out = [np.argmax(hyp[0, i, :]) for i in range(timesteps)]
            no_hidden, lr = hidden_layers[out[0]], learning_rates[out[1]]
            val_acc = train_network(no_hidden, lr)
            output = "\nController Loss : {}\n".format(sess.run(loss, feed_dict={inp: hyp}))
            output += "Accuracy : {}, Learning Rate : {}, Hidden Number : {}\n".format(val_acc, lr, no_hidden)
            with open("accuracy.log", "a+") as f:
                f.write(output)
            print(output)
            hyp = np.roll(hyp, 1, axis=1)
            _ = sess.run(train, feed_dict = {val_accuracy : val_acc ** 3, inp:hyp})
            hyp = sess.run(hyperparams, feed_dict={inp : hyp})
            # Remove last one and pad the start by 1

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