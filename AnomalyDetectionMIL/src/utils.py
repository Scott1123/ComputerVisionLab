import os
import glob

import numpy as np
from keras import backend as K

# train file path
TRAIN_DATA_DIR = '/data/UCF_Anomaly_Dataset/C3D_Features/Train/'
OUTPUT_DIR = '/data/UCF_Anomaly_Dataset/Trained_Models/TrainedModel_MIL_C3D/'
FINAL_MODEL_PATH = OUTPUT_DIR + 'final_model.hdf5'

# test file path
# C3D features(txt file) of each video. Each file contains 32 features, each of 4096 dimensions.
TEST_DATA_DIR = '/data/UCF_Anomaly_Dataset/C3D_Complete_Video_txt/Test/'
# the folder where you can save your results
RESULTS_DIR = '../Eval_Res/'
MODEL_DIR = '../Trained_AnomalyModel/'
MODEL_PATH = MODEL_DIR + 'final_model.hdf5'

# parameters
num_seg = 32
batch_size = 60
num_feat = num_seg * batch_size


def custom_objective(y_true, y_pred):
    # hyper_parameter
    lambda_1 = 0.0008  # for temporal_smoothness_term
    lambda_2 = 0.0008  # for sparsity_term

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
        tmp_true = y_true[i * num_seg: i * num_seg + num_seg]
        sum_true = K.concatenate([sum_true, K.stack(K.sum(tmp_true))])

        # init sum_pred, max_pred
        tmp_pred = y_pred[i * num_seg: i * num_seg + num_seg]
        sum_pred = K.concatenate([sum_pred, K.stack(K.sum(tmp_pred))])
        max_pred = K.concatenate([max_pred, K.stack(K.max(tmp_pred))])

        # calculate temporal_smoothness_term
        vec = K.ones_like(tmp_pred)
        v_n0 = K.concatenate([vec, tmp_pred])[(num_seg - 1):]
        v_n1 = K.concatenate([tmp_pred, vec])[:(num_seg + 1)]
        dif = (v_n0 - v_n1)[1:num_seg]
        dif2 = K.sum(K.pow(dif, 2))
        pow_dif_pred = K.concatenate([pow_dif_pred, K.stack(dif2)])

    preds = max_pred[num_feat:]
    trues = sum_true[num_feat:]

    sparsity_term = sum_pred[num_feat:(num_feat + batch_size / 2)]
    temporal_smoothness_term = pow_dif_pred[num_feat:(num_feat + batch_size / 2)]

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

    loss = loss[num_feat:]
    loss = K.mean(loss, axis=-1) + lambda_1 * temporal_smoothness_term + lambda_2 * sparsity_term

    return loss


def load_train_data_batch(abnormal_path, normal_path):
    print("0. Loading training batch")

    n_exp = batch_size / 2  # Number of abnormal and normal videos

    num_abnormal = 810  # Total number of abnormal videos in Training Dataset.
    num_normal = 800  # Total number of Normal videos in Training Dataset.

    # the features of abnormal videos and normal videos are located in two different folders.
    # get indexes for randomly selected abnormal and normal videos
    abnormal_list_iter = np.random.permutation(num_abnormal)
    abnormal_list_iter = abnormal_list_iter[num_abnormal - n_exp:]
    normal_list_iter = np.random.permutation(num_normal)
    normal_list_iter = normal_list_iter[num_normal - n_exp:]

    all_videos_path = abnormal_path

    def _listdir_nohidden(all_videos_path):
        file_dir_extension = os.path.join(all_videos_path, '*_C.txt')
        for f in glob.glob(file_dir_extension):
            if not f.startswith('.'):
                yield os.path.basename(f)

    all_videos = sorted(_listdir_nohidden(all_videos_path))
    all_videos.sort()
    all_features = []  # To store C3D features of a batch
    print("1. Loading Abnormal videos Features...")

    cnt_video = -1
    for i in abnormal_list_iter:
        cnt_video = cnt_video + 1
        video_path = os.path.join(all_videos_path, all_videos[i])
        f = open(video_path, "r")
        words = f.read().split()
        num_feat = len(words) // 4096  # 32

        count = -1
        video_features = []
        for feat in range(num_feat):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                video_features = feat_row1
            if count > 0:
                video_features = np.vstack((video_features, feat_row1))

        if cnt_video == 0:
            all_features = video_features
        if cnt_video > 0:
            all_features = np.vstack((all_features, video_features))

    print("Abnormal Features loaded.")

    print("2. Loading Normal videos...")
    all_videos_path = normal_path

    all_videos = sorted(_listdir_nohidden(all_videos_path))
    all_videos.sort()

    for i in normal_list_iter:
        video_path = os.path.join(all_videos_path, all_videos[i])
        f = open(video_path, "r")
        words = f.read().split()
        num_feat = len(words) // 4096  # 32

        count = -1
        video_features = []
        for feat in range(0, num_feat):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                video_features = feat_row1
            if count > 0:
                video_features = np.vstack((video_features, feat_row1))

        all_features = np.vstack((all_features, video_features))

    print("Normal Features loaded.")

    print("3. Loading labels...")
    num = batch_size * 32
    all_labels = [0 if i < num / 2 else 1 for i in range(num)]
    print("Labels loaded.")

    return all_features, all_labels


def load_test_data_one_video(test_video_path):
    video_path = test_video_path
    f = open(video_path, "r")
    words = f.read().split()
    num_feat = len(words) // 4096  # 32

    count = -1
    video_features = []
    for feat in range(num_feat):
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        count = count + 1
        if count == 0:
            video_features = feat_row1
        if count > 0:
            video_features = np.vstack((video_features, feat_row1))
    all_features = video_features

    return all_features
