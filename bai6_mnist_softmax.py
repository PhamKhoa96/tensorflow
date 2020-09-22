# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:14:55 2020

@author: phamk
"""


import warnings

warnings.filterwarnings('ignore')

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Data loaddings

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

X_train = mnist.train.images
X_test = mnist.test.images
X_val = mnist.validation.images

y_train = mnist.train.labels
y_test = mnist.test.labels
y_val = mnist.validation.labels

X_train.shape

y_train[0]

# Placeholder
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Variable
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Model hyper parameters
learning_rate = 0.01
batch_size = 128
nb_epochs = 100

# Define graph
logits = tf.matmul(X, W) + b
y_pred = tf.nn.softmax(logits=logits)

# Loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(entropy)
correct_preds = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

# Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Init variables
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

# Training
nb_batch = X_train.shape[0] // batch_size

for i in range(nb_epochs):
    for _ in range(nb_batch):
        X_batch , y_batch = mnist.train.next_batch(batch_size=batch_size)
        _, batch_loss = sess.run([optimizer, loss], feed_dict={X : X_batch, y : y_batch})
        
    if i % 10 == 0:
        _, val_loss, val_accuracy = sess.run([optimizer, loss, accuracy], feed_dict={X : X_val, y : y_val})
        print("Epochs {} val_loss = {} val_accuracy = {}".format(i, val_loss, val_accuracy))
        
_, test_loss, test_accuracy = sess.run([optimizer, loss, accuracy], feed_dict={X : X_test, y : y_test})
print("Epochs {} test_loss = {} test_accuracy = {}".format(i, test_loss, test_accuracy))

import matplotlib.pyplot as plt

def show_result(X_true, y_true):
    plt.imshow(X_true.reshape(28, 28))
    print(np.argmax(y_true))
    y_preds = sess.run(y_pred, feed_dict={X : [X_true]})
    
    print(np.argmax(y_preds))
    
show_result(X_test[200], y_test[200])