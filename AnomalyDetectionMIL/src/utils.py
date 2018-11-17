import os
import glob

import numpy as np
from keras import backend as K

# file path
TRAIN_DATA_DIR = '/data/UCF_Anomaly_Dataset/C3D_Features/Train/'
OUTPUT_DIR = '/data/UCF_Anomaly_Dataset/Trained_Models/TrainedModel_MIL_C3D/'
FINAL_MODEL_PATH = OUTPUT_DIR + 'final_model.hdf5'

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
    #    print("Loading training batch")

    n_exp = batch_size / 2  # Number of abnormal and normal videos

    Num_abnormal = 810  # Total number of abnormal videos in Training Dataset.
    Num_Normal = 800  # Total number of Normal videos in Training Dataset.

    # We assume the features of abnormal videos and normal videos are located in two different folders.
    Abnor_list_iter = np.random.permutation(Num_abnormal)
    Abnor_list_iter = Abnor_list_iter[Num_abnormal - n_exp:]  # Indexes for randomly selected Abnormal Videos
    Norm_list_iter = np.random.permutation(Num_Normal)
    Norm_list_iter = Norm_list_iter[Num_Normal - n_exp:]  # Indexes for randomly selected Normal Videos

    AllVideos_Path = abnormal_path

    def listdir_nohidden(AllVideos_Path):  # To ignore hidden files
        file_dir_extension = os.path.join(AllVideos_Path, '*_C.txt')
        for f in glob.glob(file_dir_extension):
            if not f.startswith('.'):
                yield os.path.basename(f)

    All_Videos = sorted(listdir_nohidden(AllVideos_Path))
    All_Videos.sort()
    AllFeatures = []  # To store C3D features of a batch
    print("Loading Abnormal videos Features...")

    Video_count = -1
    for iv in Abnor_list_iter:
        Video_count = Video_count + 1
        VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        num_feat = len(words) / 4096
        # Number of features per video to be loaded.
        # In our case num_feat=32, as we divide the video into 32 segments.

        # Note that:
        # we have already computed C3D features for the whole video and divide the video features
        # into 32 segments. Please see Save_C3DFeatures_32Segments.m as well

        count = -1
        VideoFeatues = []
        for feat in range(num_feat):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))

        if Video_count == 0:
            AllFeatures = VideoFeatues
        if Video_count > 0:
            AllFeatures = np.vstack((AllFeatures, VideoFeatues))
        print(" Abnormal Features  loaded")

    print("Loading Normal videos...")
    AllVideos_Path = normal_path

    def listdir_nohidden(AllVideos_Path):  # To ignore hidden files
        file_dir_extension = os.path.join(AllVideos_Path, '*_C.txt')
        for f in glob.glob(file_dir_extension):
            if not f.startswith('.'):
                yield os.path.basename(f)

    All_Videos = sorted(listdir_nohidden(AllVideos_Path))
    All_Videos.sort()

    for iv in Norm_list_iter:
        VideoPath = os.path.join(AllVideos_Path, All_Videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        feat_row1 = np.array([])
        num_feat = len(
            words) / 4096  # Number of features to be loaded. In our case num_feat=32, as we divide the video into 32 segments.

        count = -1
        VideoFeatues = []
        for feat in range(0, num_feat):

            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                VideoFeatues = feat_row1
            if count > 0:
                VideoFeatues = np.vstack((VideoFeatues, feat_row1))
            feat_row1 = []
        AllFeatures = np.vstack((AllFeatures, VideoFeatues))

    print("Features  loaded")

    # load labels
    AllLabels = np.zeros(32 * batch_size, dtype='uint8')
    th_loop1 = n_exp * 32
    th_loop2 = n_exp * 32 - 1

    for iv in range(0, 32 * batch_size):
        if iv < th_loop1:
            AllLabels[iv] = int(0)  # abnormal videos are labeled 0.
        if iv > th_loop2:
            AllLabels[iv] = int(1)  # normal videos are labeled 1.

    print("ALLabels  loaded")

    return AllFeatures, AllLabels
