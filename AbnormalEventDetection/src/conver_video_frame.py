# coding=utf-8
import numpy as np
import os
from scipy.misc import imresize


def conver_video_frame(video_source_path, source_frame_path, fps):
    imagestore = []
    videos = os.listdir(video_source_path)
    for num, video in enumerate(videos[:]):
        os.mkdir(source_frame_path + r'\frames{}'.format(str(num + 1)))
        framepath = source_frame_path + r'\frames{}'.format(str(num + 1))
        os.system(
            r'ffmpeg -i {}/{} -r {}  {}\frames{}\%03d.jpg'.format(video_source_path, video, fps, source_frame_path,
                                                                  str(num + 1)))
        images = os.listdir(framepath)
        for image in images:
            image_path = framepath + '\\' + image


if __name__ == '__main__':
    project_path = os.path.dirname(os.path.realpath(__file__))
    conver_video_frame(project_path + r"\all_test_video", project_path + r"\all_test_frames", 10)
