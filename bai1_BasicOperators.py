# -*- coding: utf-8 -*-
"""
Created on Thu May 21 08:20:36 2020

@author: phamk
"""

#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# In tensorflow version
print(tf.__version__)

x1 = tf.constant(1, name='const_x1')
print(x1)

x2 = tf.constant(2.)
print(x2)

x3 = tf.constant(3, dtype='float32')
print(x3)

v1 = tf.constant([1, 2, 3], dtype=tf.float32, name='v1_const_vector')
print(v1)
sess = tf.compat.v1.Session()
print(sess.run(v1))

# Create tensor 2D & use session to show value
v2 = tf.constant([
    [1, 2, 3],
    [3, 4, 5]
], dtype=tf.float32, name='2d_matrix')
print(sess.run(v2))
print(v2.eval(session=sess))


#Perform operators
y1 = x2 + x3

print(y1)
print(sess.run(y1))

v3 = tf.constant([
    [1, 1, 1],
    [3, 3, 2]
], dtype=tf.float32, name='2d_matrix')

y2 = v2 + v3
print(y2)
print(sess.run(y2))


#Placeholder
p1 = tf.placeholder(dtype=tf.float32)
p2 = tf.placeholder(dtype=tf.float32)
# Asign operation for placeholder
o_add = p1 + p2
o_mul = p1 * p2
o_delta = p1**2 + p2
print(o_add)


#Feed value for placeholder
d_values = {
    p1 : 20,
    p2 : 10
}
print(sess.run(o_delta, feed_dict=d_values))
print(sess.run([o_add, o_mul, o_delta], feed_dict=d_values))


#Variables
var_1 = tf.Variable(name='var_1', initial_value=10)
var_2 = tf.Variable(name='var_2', initial_value=50)
#Note cần phải khởi tạo biến trước khi đưa vào tính toán trên sessionNote cần phải khởi tạo biến trước khi đưa vào tính toán trên session
sess.run(var_1.initializer)
print(sess.run(var_1))
tf.global_variables_initializer().run(session=sess)
print(sess.run(var_2))