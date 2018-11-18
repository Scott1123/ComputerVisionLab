from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Reshape
# from keras.layers import TimeDistributedDense
from keras.regularizers import l2
from keras.optimizers import SGD, adam, Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import keras.backend as T
# import theano.tensor as T
import theano
import csv
# import ConfigParser
import collections
import time
# import csv
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
import path
from os.path import basename
import glob
# import theano.sandbox

# theano.sandbox.cuda.use('gpu0')


def load_model(json_path):  # Function to load the model
    model = model_from_json(open(json_path).read())
    return model


def load_weights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model


def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


def save_model(model, json_path, weight_path):  # Function to save the model
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)
