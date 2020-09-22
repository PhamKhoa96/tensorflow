# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:22:32 2020

@author: phamk
"""


#import tensorflow as tf

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import os

import tensorflow as tf

import cProfile
print(tf.__version__)

tf.config.experimental_run_functions_eagerly(True)
#tf.enable_eager_execution()
print(tf.executing_eagerly())

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))