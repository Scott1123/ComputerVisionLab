# !/usr/bin/python
# -*-coding:utf-8-*-

import os
import numpy as np

# train file path
TRAIN_DATA_DIR = 'E:/github_projects/data/UCF_Anomaly_Dataset/C3D_Features/Train/'
OUTPUT_DIR = '../model/'
MODEL_PATH = OUTPUT_DIR + 'model.h5'

# test file path
# C3D features(txt file) of each video. Each file contains 32 features, each of 4096 dimensions.
TEST_DATA_DIR = 'E:/github_projects/data/UCF_Anomaly_Dataset/C3D_Features/Test/'
# the folder where you can save your results
RESULTS_DIR = '../res/'

# parameters
batch_size = 60  # train batch
one_video_seg = 32  # single video
one_video_feat = 32  # single video
one_batch_feat = one_video_seg * batch_size  # for one batch

num_abnormal = 810
num_normal = 800

for i in range(num_abnormal):
    data = np.random.randint(1000, 600000, (32, 4096))
    data = data / 1000000
    with open(TRAIN_DATA_DIR + 'abnormal_' + str(i) + '.txt', 'w') as f:
        for row in range(32):
            for col in range(4096):
                f.write('%.6f ' % data[row][col])
            f.write('\n')
    print('abnormal file %d finished.' % i)
print('abnormal data finished.')

