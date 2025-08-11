import cv2
import numpy as np
import os
import os.path as osp
from glob import glob
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path and first frame index
args = parse_args()
root_path = args.root_path

# copy inputs from root_path
cmd = 'cp ' + osp.join(root_path, 'img_to_anim.png') + ' .'
os.system(cmd)
cmd = 'cp ' + osp.join(root_path, 'video_to_anim.mp4') + ' .'
os.system(cmd)

# run MimicMotion
cmd = 'python inference.py --inference_config configs/test.yaml'
os.system(cmd)

# save animated video to root_path
cmd = 'mv output.mp4 ' + osp.join(root_path, 'video.mp4')
os.system(cmd)

# extract frames from the video
os.makedirs(osp.join(root_path, 'images'), exist_ok=True)
os.system('rm -rf ' + osp.join(root_path, 'images', '*'))
vidcap = cv2.VideoCapture(osp.join(root_path, 'video.mp4'))
frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
success, frame = vidcap.read()
frame_idx = 0
while success:
    print(str(frame_idx) + '/' + str(frame_num), end='\r')
    cv2.imwrite(osp.join(root_path, 'images', str(frame_idx) + '.png'), frame)
    success, frame = vidcap.read()
    frame_idx += 1


