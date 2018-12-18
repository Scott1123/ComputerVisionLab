# !/usr/bin/python
# -*-coding:utf-8-*-
import os
import shutil


UCF_ROOT = '/home/timchen/media/UCF_Crimes/'
VIDEO_DIR = os.path.join(UCF_ROOT, 'Videos')
CSV_DIR = os.path.join(UCF_ROOT, 'C3D_Features_csv')


def mkdir():
    if os.path.exists(CSV_DIR):
        shutil.rmtree(CSV_DIR)
    os.makedirs(CSV_DIR)
    print('make <CSV_DIR>... OK')

    folder_list = os.listdir(VIDEO_DIR)
    folder_list = [name for name in folder_list if name != 'Normal_Videos_event']
    for folder in folder_list:
        os.makedirs(os.path.join(CSV_DIR, folder))
        videos = os.listdir(os.path.join(VIDEO_DIR, folder))
        for video in videos:
            # print(type(video))
            video_folder = os.path.join(CSV_DIR, folder, video)
            os.makedirs(video_folder)

    print('new csv folders... OK')


if __name__ == '__main__':
    mkdir()
