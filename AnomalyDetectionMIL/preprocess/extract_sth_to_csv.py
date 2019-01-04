# !/usr/bin/python
# -*-coding:utf-8-*-
import os
import time
import subprocess as sp

C3D_EXTRACT_ROOT = '/home/timchen/c3d/C3D/C3D-v1.0/examples/c3d_feature_extraction'
EXTRACTOR = os.path.join(C3D_EXTRACT_ROOT, 'extract_2019.py')

UCF_ROOT = '/home/timchen/media/UCF_Crimes/'
ALL_LIST = os.path.join(UCF_ROOT, 'Anomaly_Detection_splits', 'Anomaly_All.txt')
VIDEO_DIR = os.path.join(UCF_ROOT, 'Videos')
CSV_OUT_DIR = os.path.join(UCF_ROOT, 'C3D_Features_csv')
# LOG_FILE = os.path.join(UCF_ROOT, 'logs', 'log_20181228_2.txt')
needed_list = ['Training_Normal_Videos_Anomaly']


def extract(video_list_file, video_dir, out_dir, extractor):
    print('start extract all videos...')

    with open(video_list_file, 'r') as f:
        file_list = f.read().splitlines()
        for file_ in file_list:
            event_type = file_.split('/')[0]
            cur_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
            print('[current time: %s]' % cur_time)
            if event_type not in needed_list:
                print('%s has been extracted, skip...' % file_)
                continue
            print('extract %s...' % file_)
            input_ = os.path.join(video_dir, file_)
            output_ = os.path.join(out_dir, file_)
            # cmd = 'timpy27 %s %s %s %s' % (extractor, input_, output_, file_)
            cmd_list = ['timpy27', extractor, input_, output_, file_]
            print('cmd: ', ' '.join(cmd_list))
            res = sp.check_call(cmd_list)
            print('extract ' + file_ + '... OK')

    print('extract all videos... OK')


if __name__ == '__main__':
    extract(ALL_LIST, VIDEO_DIR, CSV_OUT_DIR, EXTRACTOR)

