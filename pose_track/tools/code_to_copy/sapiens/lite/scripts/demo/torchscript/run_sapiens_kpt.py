import os
import os.path as osp
from glob import glob
import json
import numpy as np
import cv2
import argparse

def make_video(video_output_path, output_path):
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(output_path, '*.png'))])
    img_height, img_width = cv2.imread(osp.join(output_path, str(frame_idx_list[0]) + '.png')).shape[:2]
    video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width, img_height))
    for frame_idx in frame_idx_list:
        frame = cv2.imread(osp.join(output_path, str(frame_idx) + '.png'))
        frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
        video.write(frame.astype(np.uint8))
    video.release()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--vis_format', type=str, dest='vis_format')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    assert args.vis_format, "Please set vis_format."
    return args

args = parse_args()
root_path = args.root_path
vis_format = args.vis_format
input_path = './input'
output_path = './output_pose'
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# prepare input images
os.system('rm -rf ' + osp.join(input_path, '*'))
os.system('cp ' + osp.join(root_path, 'images', '*') + ' ' + osp.join(input_path, '.'))
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(input_path, '*.png'))])

# pose
os.system('rm -rf ' + osp.join(output_path, '*'))
os.system('./pose_keypoints133.sh')
os.makedirs(osp.join(root_path, 'keypoints_whole_body'), exist_ok=True)
for frame_idx in frame_idx_list:
    with open(osp.join(output_path, str(frame_idx) + '.json')) as f:
        kpt = json.load(f)['instance_info'][0]
    kpt, score = np.array(kpt['keypoints'], dtype=np.float32), np.array(kpt['keypoint_scores'], dtype=np.float32)[:,None]
    kpt = np.concatenate((kpt, score),1)
    with open(osp.join(root_path, 'keypoints_whole_body', str(frame_idx) + '.json'), 'w') as f:
        json.dump(kpt.tolist(), f)
if vis_format == 'image':
    os.system('mv ' + osp.join(output_path, '*.png') + ' ' + osp.join(root_path, 'keypoints_whole_body', '.'))
else:
    make_video(osp.join(root_path, 'keypoints_whole_body.mp4'), output_path)

