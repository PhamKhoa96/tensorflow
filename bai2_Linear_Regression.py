# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:38:15 2020

@author: phamk
"""


#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

sess = tf.compat.v1.Session()

# X = [1, 2, 3, 4] <=> [x0, x1, x2, x3]
# Y = [2] <=> w_init + w_0*x_0 + ... + w_3*x3

# Fake X data
X_data = np.random.random((10000, 2))

# Fake sample weights
sample_weights = np.array([3, 4]).reshape(2, )

# Fake y_data
y_data = np.matmul(X_data, sample_weights)

# Approximation Y
y_data = np.add(y_data, np.random.uniform(-0.5, 0.5))
y_data = y_data.reshape(len(y_data), 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

n_dim = X_train.shape[1]
print(n_dim)
# Placeholder for pass data
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# Trainable weights
W = tf.Variable(tf.ones([n_dim, 1]))
b = tf.Variable(np.random.randn(), dtype=tf.float32)
tf.global_variables_initializer().run(session=sess)
print(sess.run(W))
print(sess.run(b))

pred = tf.add(tf.matmul(X, W), b)

loss = tf.reduce_mean(tf.square(pred - Y))

learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

# Training

sess.run(init)

epochs = 5000

loss_history = []

for epoch in range(epochs):
    sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
    
    test_loss = sess.run(loss, feed_dict={X : X_test, Y: y_test})
    
    loss_history.append(test_loss)
    
    if epoch % 500 == 0:
        print("Epoch {} Test loss = {}".format(epoch, test_loss))
        
print("Training finished")
print(sess.run(W))
print(sess.run(b))


import matplotlib.pyplot as plt

plt.plot(range(len(loss_history)), loss_history)
plt.axis([0, epochs, 0, np.max(loss_history)])
plt.show()

pred_y = sess.run(pred, feed_dict={
    X : [[3, 10]]
})

print(pred_y)