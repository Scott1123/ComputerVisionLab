import datetime

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

model.compile(loss=custom_objective, optimizer=Adagrad(lr=0.01, epsilon=1e-08))


# 2. train model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

All_class_files = os.listdir(TRAIN_DATA_DIR)
All_class_files.sort()
num_iters = 8000  # 20000
total_iterations = 0
tmp_start = datetime.now()

for it_num in range(num_iters):

    AbnormalPath = os.path.join(TRAIN_DATA_DIR, All_class_files[0])
    NormalPath = os.path.join(TRAIN_DATA_DIR, All_class_files[1])
    inputs, targets = load_train_data_batch(AbnormalPath, NormalPath)
    batch_loss = model.train_on_batch(inputs, targets)
    total_iterations += 1
    if total_iterations % 20 == 1:
        print("These iteration=" +
              str(total_iterations) + ") took: " +
              str(datetime.now() - tmp_start) +
              ", with batch loss of " + str(batch_loss))
    if total_iterations % 1000 == 0:  # Save the model at every 1000th iterations.
        tmp_model_path = OUTPUT_DIR + 'tmp_model_' + str(total_iterations) + '.hdf5'
        model.save(tmp_model_path)

model.save(FINAL_MODEL_PATH)

