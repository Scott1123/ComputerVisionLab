# !/usr/bin/python
# -*-coding:utf-8-*-
import os
import glob
import numpy as np

# parameters
batch_size = 60  # train batch
one_video_seg = 32  # single video
one_video_feat = 32  # single video
one_batch_feat = one_video_seg * batch_size  # for one batch


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
