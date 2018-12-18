import os
import glob

import numpy as np
import tensorflow as tf
import keras.backend as K

# train file path
# TRAIN_DATA_DIR = '/data/UCF_Anomaly_Dataset/C3D_Features/Train/'
TRAIN_DATA_DIR = '/home/timchen/AnomalyDetectionCVPR2018/SampleVideos/'
# OUTPUT_DIR = '../model/'
OUTPUT_DIR = '/home/timchen/AnomalyDetectionCVPR2018/output/'
MODEL_PATH = OUTPUT_DIR + 'model.h5'

# test file path
# C3D features(txt file) of each video. Each file contains 32 features, each of 4096 dimensions.
TEST_DATA_DIR = '/data/UCF_Anomaly_Dataset/C3D_Features/Test/'
# the folder where you can save your results
RESULTS_DIR = '../res/'

# parameters
batch_size = 6  # train batch
one_video_seg = 32  # single video
one_video_feat = 32  # single video
one_batch_feat = one_video_seg * batch_size  # for one batch

num_abnormal = 6
num_normal = 6

# hyper_parameter
lambda_1 = 0.0008  # for temporal_smoothness_term
lambda_2 = 0.0008  # for sparsity_term
data_type = tf.float64


def custom_loss(y_true, y_pred):
    # init
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    # terms
    sum_true = K.variable([], dtype=data_type)
    sum_pred = K.variable([], dtype=data_type)  # sparsity term
    max_pred = K.variable([], dtype=data_type)
    pow_dif_pred = K.variable([], dtype=data_type)  # temporal smoothness term

    for i in range(batch_size):
        # init sum_true
        tmp_true = y_true[i * one_video_seg: i * one_video_seg + one_video_seg]
        sum_true = K.concatenate([sum_true, [K.sum(tmp_true, axis=-1)]])

        # init sum_pred, max_pred
        tmp_pred = y_pred[i * one_video_seg: i * one_video_seg + one_video_seg]
        sum_pred = K.concatenate([sum_pred, [K.sum(tmp_pred, axis=-1)]])
        max_pred = K.concatenate([max_pred, [K.max(tmp_pred, axis=-1)]])

        # calculate temporal_smoothness_term
        # dif = [tmp_pred[k] - tmp_pred[k + 1] for k in range(one_video_seg - 1)]
        # print(type(tmp_pred))
        one = K.ones_like(tmp_pred)
        v0 = K.concatenate([one, tmp_pred])
        v1 = K.concatenate([tmp_pred, one])
        dif = (v1[:one_video_seg+1] - v0[one_video_seg-1:])[1:]
        dif = K.concatenate([dif, [tmp_pred[one_video_seg - 1] - 1]])
        pow_dif_pred = K.concatenate([pow_dif_pred, [K.sum(K.pow(dif, 2))]])

    preds = max_pred
    trues = sum_true

    sparsity_term = K.sum(sum_pred, axis=-1)  # 0:batch_size//2 ?
    temporal_smoothness_term = K.sum(pow_dif_pred, axis=-1)  # 0:batch_size//2 ?

    # get normal & abnormal preds
    normal_pred = tf.boolean_mask(preds, K.equal(trues, one_video_seg))
    abnormal_pred = tf.boolean_mask(preds, K.equal(trues, 0))

    loss = K.variable([], dtype=data_type)
    for i in range(batch_size // 2):
        p0 = K.maximum(K.cast(0, dtype=data_type), 1 - abnormal_pred[i] + normal_pred[i])
        # print(i, K.eval(p0))
        loss = K.concatenate([loss, [p0]])

    loss = tf.reduce_mean(loss) + lambda_1 * temporal_smoothness_term + lambda_2 * sparsity_term

    return loss


def load_train_data_batch(abnormal_path, normal_path):

    index_abnormal = np.random.permutation(num_abnormal)[:(batch_size // 2)]
    index_normal = np.random.permutation(num_normal)[:(batch_size // 2)]

    video_abnormal = sorted(_listdir_nohidden(abnormal_path))
    video_normal = sorted(_listdir_nohidden(normal_path))

    all_features = []

    print('1. loading abnormal video features...')
    for idx in index_abnormal:
        video_path = os.path.join(abnormal_path, video_abnormal[idx])
        f = open(video_path, 'r')
        content = f.read().split()

        for i in range(one_video_feat):
            tmp_feat = np.float32(content[i * 4096:i * 4096 + 4096])
            if len(all_features) == 0:
                all_features = tmp_feat
            else:
                all_features = np.vstack((all_features, tmp_feat))

    print('3. loading normal video features...')
    for idx in index_normal:
        video_path = os.path.join(normal_path, video_normal[idx])
        f = open(video_path, 'r')
        content = f.read().split()

        for i in range(one_video_feat):
            tmp_feat = np.float32(content[i * 4096:i * 4096 + 4096])
            if len(all_features) == 0:
                all_features = tmp_feat
            else:
                all_features = np.vstack((all_features, tmp_feat))

    print("3. loading labels...")
    num = batch_size * 32
    all_labels = [0 if i < num / 2 else 1 for i in range(num)]

    print("batch TRAIN data loaded.")

    return all_features, all_labels


def load_test_data_one_video(test_video_path):
    all_features = []

    f = open(test_video_path, 'r')
    content = f.read().split()

    for i in range(one_video_feat):
        tmp_feat = np.float32(content[i * 4096:i * 4096 + 4096])
        if len(all_features) == 0:
            all_features = tmp_feat
        else:
            all_features = np.vstack((all_features, tmp_feat))

    return all_features


def _listdir_nohidden(all_videos_path):
    file_dir_extension = os.path.join(all_videos_path, '*264.txt')
    for f in glob.glob(file_dir_extension):
        if not f.startswith('.'):
            yield os.path.basename(f)

