# !/usr/bin/python
# -*-coding:utf-8-*-
# import numpy as np
# import keras.backend as K
# import tensorflow as tf

#
# x = np.array([1, 2, 5, 5, 4, 3, 5, 5, 10, 9])
# y = np.array([18, 28, 58, 582, 48, 38, 58, 58, 108, 98])
#
# a = tf.greater_equal(x, 5)
#
# b = tf.boolean_mask(y, a)
#
# print(b)
# print(K.eval(b))

array = [1, 8, 15]
g = (x for x in array if array.count(x) > 0)
# array = [2, 8, 22]

print(type(g))

print(list(g))
