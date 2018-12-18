# !/usr/bin/python
# -*-coding:utf-8-*-
import os
import csv
import shutil

UCF_ROOT = '/home/timchen/media/UCF_Crimes/'

CSV_DIR = os.path.join(UCF_ROOT, 'C3D_Features_csv')
TXT_DIR = os.path.join(UCF_ROOT, 'C3D_Features_txt')

ALL_DATA = os.path.join(UCF_ROOT, 'All_Data')
TRAIN_NORMAL = os.path.join(ALL_DATA, 'Train_Data', 'Normal_Videos')
TRAIN_ABNORMAL = os.path.join(ALL_DATA, 'Train_Data', 'Abnormal_Videos')
TEST_NORMAL = os.path.join(ALL_DATA, 'Test_Data', 'Normal_Videos')
TEST_ABNORMAL = os.path.join(ALL_DATA, 'Test_Data', 'Abnormal_Videos')

TRAIN_LIST = os.path.join(UCF_ROOT, 'Anomaly_Detection_splits', 'Anomaly_Train.txt')
TEST_LIST = os.path.join(UCF_ROOT, 'Anomaly_Detection_splits', 'Anomaly_Test.txt')


def merge_single_video(video_name, csv_folder, txt_file):
    num_clips = len(os.listdir(csv_folder))
    # print(csv_folder + ' clips: ' + str(num_clips))

    seg_size = num_clips // 32

    f_output = open(txt_file, 'w')

    for i in range(32):
        data = [float(0) for k in range(4096)]
        for j in range(i * seg_size, (i + 1) * seg_size):
            clip_name = (video_name + str('%06d' % (j * 16)) + '.csv')
            with open(os.path.join(csv_folder, clip_name), 'r') as f:
                reader = csv.reader(f)
                line = next(reader)
                line = [float(x) for x in line]
                for m in range(4096):
                    data[m] += line[m]
        data = [('%.6f' % (x / seg_size)) for x in data]
        data = ' '.join(data)
        f_output.write(data + ' \n')

    f_output.close()
    print('merge ' + video_name + ' to txt... OK')


def merge_all(txt_dir, csv_dir):
    if os.path.exists(txt_dir):
        shutil.rmtree(txt_dir)
    os.makedirs(txt_dir)
    print('make <txt_dir>... OK')

    folder_list = os.listdir(csv_dir)
    for folder in folder_list:
        os.makedirs(os.path.join(txt_dir, folder))
        videos = os.listdir(os.path.join(csv_dir, folder))
        for video in videos:
            csv_folder = os.path.join(csv_dir, folder, video)
            txt_file = os.path.join(txt_dir, folder, video + '.txt')
            merge_single_video(video, csv_folder, txt_file)
        print('merge folder: ' + folder + ' successfully.')

    print('merge all... OK')


def split_files(src_dir, normal_folder):
    # init all data folder
    if os.path.exists(ALL_DATA):
        shutil.rmtree(ALL_DATA)
    os.makedirs(TRAIN_NORMAL)
    os.makedirs(TRAIN_ABNORMAL)
    os.makedirs(TEST_NORMAL)
    os.makedirs(TEST_ABNORMAL)
    print('make <all_data>... OK')

    # train files
    with open(TRAIN_LIST, 'r') as f:
        train_files = f.read().splitlines()

    for file in train_files:
        folder, name = file.split('/')
        if folder in normal_folder:
            shutil.copyfile(os.path.join(src_dir, file), TRAIN_NORMAL)
        else:
            shutil.copyfile(os.path.join(src_dir, file), TRAIN_ABNORMAL)
    print('copy train files... OK')

    # test files
    with open(TEST_LIST, 'r') as f:
        test_files = f.read().splitlines()

    for file in test_files:
        folder, name = file.split('/')
        if folder in normal_folder:
            shutil.copyfile(os.path.join(src_dir, file), TEST_NORMAL)
        else:
            shutil.copyfile(os.path.join(src_dir, file), TEST_ABNORMAL)
    print('copy test files... OK')


if __name__ == '__main__':
    print('start merge...')
    merge_all(TXT_DIR, CSV_DIR)
    normal_folders = ['Testing_Normal_Videos_Anomaly', 'Testing_Normal_Videos_Anomaly']
    # abnormal_folders = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents',
    #                     'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
    split_files(TXT_DIR, normal_folders)
