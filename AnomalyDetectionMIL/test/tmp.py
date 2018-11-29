# !/usr/bin/python
# -*-coding:utf-8-*-
import numpy as np
import keras.backend as K
import tensorflow as tf


x = np.array([1, 2, 5, 5, 4, 3, 5, 5, 10, 9])
y = np.array([18, 28, 58, 582, 48, 38, 58, 58, 108, 98])
print(x)
res = tf.Variable(x)
ad = K.sum(3)
print(ad)
res = K.concatenate([res, [ad]])

print(res)
print(K.eval(res))
