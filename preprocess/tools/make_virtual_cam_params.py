import os
import os.path as osp
import json
import cv2
from glob import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path

save_root_path = osp.join(root_path, 'cam_params')
os.makedirs(save_root_path, exist_ok=True)
frame_idx_list = [int(x.split('/')[-1][:-4]) for x in glob(osp.join(root_path, 'images', '*.png'))]
for frame_idx in frame_idx_list:
    img_height, img_width = cv2.imread(osp.join(root_path, 'images', str(frame_idx) + '.png')).shape[:2]
    with open(osp.join(save_root_path, str(frame_idx) + '.json'), 'w') as f:
        json.dump({'R': np.eye(3).astype(np.float32).tolist(), 't': np.zeros((3), dtype=np.float32).tolist(), 'focal': (2000,2000), 'princpt': (img_width/2, img_height/2)}, f)

