from datetime import datetime

from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers import Dense, Dropout

from utils import *


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
num_iters = 1000  # 20000
total_iterations = 0
tmp_start = datetime.now()

for it_num in range(num_iters):

    abnormal_path = os.path.join(TRAIN_DATA_DIR, all_train_file[0])
    normal_path = os.path.join(TRAIN_DATA_DIR, all_train_file[1])
    inputs, targets = load_train_data_batch(abnormal_path, normal_path)
    batch_loss = model.train_on_batch(inputs, targets)
    total_iterations += 1
    if total_iterations % 20 == 1:
        print("These iteration=" +
              str(total_iterations) + ") cost: " +
              str((datetime.now() - tmp_start).seconds) +
              "s, with batch loss of " + str(batch_loss))
    if total_iterations % 1000 == 0:  # Save the model at every 1000th iterations.
        tmp_model_path = OUTPUT_DIR + 'tmp_model_' + str(total_iterations) + '.h5'
        model.save(tmp_model_path)

model.save(MODEL_PATH)
