from datetime import datetime
import time
from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers import Dense, Dropout
import tensorflow as tf
tf.python.control_flow_ops = tf

from utils import *

# determine GPU number
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 1. make model

model = Sequential()
model.add(Dense(512, input_dim=4096, init='glorot_normal', W_regularizer=l2(0.001), activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32, init='glorot_normal', W_regularizer=l2(0.001)))
model.add(Dropout(0.6))
model.add(Dense(1, init='glorot_normal', W_regularizer=l2(0.001), activation='sigmoid'))

model.compile(loss=custom_loss, optimizer=Adagrad(lr=0.01, epsilon=1e-08))


# 2. train model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

all_train_file = os.listdir(TRAIN_DATA_DIR)
all_train_file.sort()
num_iters = 4396  # 20000 3600 1000  20  3
total_iterations = 0
tmp_start = datetime.now()

abnormal_path = os.path.join(TRAIN_DATA_DIR, all_train_file[0])
normal_path = os.path.join(TRAIN_DATA_DIR, all_train_file[1])

# cur_time = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
# log_file = ('log_%s_%d.txt' % (cur_time, num_iters))
# f_log = open(os.path.join('log', log_file), 'w')

for it_num in range(num_iters):
    print('[iteration] = (' + str(it_num+1) + ')...')
    inputs, targets = load_train_data_batch(abnormal_path, normal_path)
    batch_loss = model.train_on_batch(inputs, targets)  # train on batch
    total_iterations += 1
    if total_iterations % 20 == 1:
        print("These iteration=(" +
              str(total_iterations) + ") cost: " +
              str((datetime.now() - tmp_start).seconds) +
              "s, with batch loss of " + str(batch_loss))
    if total_iterations % 1000 == 0:
        tmp_model_path = OUTPUT_DIR + 'tmp_model_' + str(total_iterations) + '.h5'
        model.save(tmp_model_path)
    # write log.
    cur_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    log_info = '[' + cur_time + '] iteration (' + str(total_iterations) + ')... OK\nloss: ' + str(batch_loss)
    print(log_info)
    # f_log.write(log_info + '\n')

# f_log.close()

# model_path = MODEL_PATH[:-3] + '_' + str(num_iters) + '.h5'
model_path = OUTPUT_DIR + 'model_' + str(num_iters) + '.h5'
model.save(model_path)
print('train finished.')
