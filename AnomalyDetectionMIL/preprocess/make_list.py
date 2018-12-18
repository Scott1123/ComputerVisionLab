# !/usr/bin/python
# -*-coding:utf-8-*-
import os


UCF_ROOT = '/home/timchen/media/UCF_Crimes/'
VIDEO_DIR = os.path.join(UCF_ROOT, 'Videos')
LIST_DIR = os.path.join(UCF_ROOT, 'Anomaly_Detection_splits')

# get video type folder list
files_name = os.listdir(VIDEO_DIR)
files_name = [name for name in files_name if name != 'Normal_Videos_event']

# get all files
all_files = []
for video_type in files_name:
    video_type_dir = os.path.join(VIDEO_DIR, video_type)
    videos = os.listdir(video_type_dir)
    for video in videos:
        name = video_type + '/' + video
        all_files.append(name)

# get test list
with open(os.path.join(LIST_DIR, 'Anomaly_Test.txt'), 'r') as f:
    test_files = f.read().splitlines()

# get train list
train_files = [name for name in all_files if name not in test_files]

# store all list and train list
with open(os.path.join(LIST_DIR, 'Anomaly_All.txt'), 'w') as f:
    for name in all_files:
        f.write(name + '\n')

with open(os.path.join(LIST_DIR, 'Anomaly_Train.txt'), 'w') as f:
    for name in all_files:
        f.write(name + '\n')

print('make list... OK')
