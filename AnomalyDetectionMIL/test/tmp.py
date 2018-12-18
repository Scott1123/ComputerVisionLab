# !/usr/bin/python
# -*-coding:utf-8-*-
import numpy as np
# import keras.backend as K
# import tensorflow as tf


def softmax(x):
    x = x - np.mean(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


a = np.array([1, 2, 3, 4, 5])
a = a * 200

print(softmax(a))
print(softmax(a - 20))

