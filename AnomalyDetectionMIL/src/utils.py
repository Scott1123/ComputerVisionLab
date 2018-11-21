import os
import glob

import numpy as np
from keras import backend as K

# train file path
TRAIN_DATA_DIR = '/data/UCF_Anomaly_Dataset/C3D_Features/Train/'
OUTPUT_DIR = '../model/'
MODEL_PATH = OUTPUT_DIR + 'model.h5'

# test file path
# C3D features(txt file) of each video. Each file contains 32 features, each of 4096 dimensions.
TEST_DATA_DIR = '/data/UCF_Anomaly_Dataset/C3D_Features/Test/'
# the folder where you can save your results
RESULTS_DIR = '../res/'

# parameters
batch_size = 60  # train batch
one_video_seg = 32  # single video
one_video_feat = 32  # single video
one_batch_feat = one_video_seg * batch_size  # for one batch

# hyper_parameter
lambda_1 = 0.0008  # for temporal_smoothness_term
lambda_2 = 0.0008  # for sparsity_term


def custom_loss(y_true, y_pred):
    # init
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    # terms
    sum_true = K.ones_like(y_true)
    sum_pred = K.ones_like(y_pred)  # sparsity term
    max_pred = K.ones_like(y_pred)
    pow_dif_pred = K.ones_like(y_pred)  # temporal smoothness term

    for i in range(batch_size):
        # init sum_true
        tmp_true = y_true[i * one_video_seg: i * one_video_seg + one_video_seg]
        sum_true = K.concatenate([sum_true, K.stack(K.sum(tmp_true))])

        # init sum_pred, max_pred
        tmp_pred = y_pred[i * one_video_seg: i * one_video_seg + one_video_seg]
        sum_pred = K.concatenate([sum_pred, K.stack(K.sum(tmp_pred))])
        max_pred = K.concatenate([max_pred, K.stack(K.max(tmp_pred))])

        # calculate temporal_smoothness_term
        vec = K.ones_like(tmp_pred)
        v_n0 = K.concatenate([vec, tmp_pred])[(one_video_seg - 1):]
        v_n1 = K.concatenate([tmp_pred, vec])[:(one_video_seg + 1)]
        dif = (v_n0 - v_n1)[1:one_video_seg]
        dif2 = K.sum(K.pow(dif, 2))
        pow_dif_pred = K.concatenate([pow_dif_pred, K.stack(dif2)])

    preds = max_pred[one_batch_feat:]
    trues = sum_true[one_batch_feat:]

    sparsity_term = sum_pred[one_batch_feat:(one_batch_feat + batch_size / 2)]
    temporal_smoothness_term = pow_dif_pred[one_batch_feat:(one_batch_feat + batch_size / 2)]

    # get index
    index_normal = K.equal(trues, 32).nonzero()[0]
    index_abnormal = K.equal(trues, 0).nonzero()[0]

    # get normal & abnormal preds
    normal_pred = preds[index_normal]
    abnormal_pred = preds[index_abnormal]

    loss = K.ones_like(y_pred)
    for i in range(batch_size // 2):
        p0 = K.max(0, 1 - abnormal_pred + normal_pred)
        loss = K.concatenate([loss, K.stack(K.sum(p0))])

    loss = loss[one_batch_feat:]
    loss = K.mean(loss, axis=-1) + lambda_1 * temporal_smoothness_term + lambda_2 * sparsity_term

    return loss


def load_train_data_batch(abnormal_path, normal_path):
    num_abnormal = 810
    num_normal = 800

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
    file_dir_extension = os.path.join(all_videos_path, '*_C.txt')
    for f in glob.glob(file_dir_extension):
        if not f.startswith('.'):
            yield os.path.basename(f)

