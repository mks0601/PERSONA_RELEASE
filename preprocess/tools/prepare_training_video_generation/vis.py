import torch
import numpy as np
import cv2
import json
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import matplotlib
from pytorch3d.io import load_ply
import argparse

import sys
cur_dir = osp.dirname(os.path.abspath(__file__))
root_dir = osp.join(cur_dir, '..', '..')
sys.path.insert(0, osp.join(root_dir, 'main'))
sys.path.insert(0, osp.join(root_dir, 'common'))
from utils.vis import render_mesh
from utils.smpl_x import smpl_x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
img_path_list = glob(osp.join(root_path, 'images', '*.png'))
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]

# render smplx meshes
os.makedirs(osp.join(root_path, 'smplx_optimized', 'renders'), exist_ok=True)
video_save = cv2.VideoWriter(osp.join(root_path, 'smplx_optimized.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
for frame_idx in tqdm(frame_idx_list):
    img = cv2.imread(osp.join(root_path, 'images', str(frame_idx) + '.png'))
    vert, _ = load_ply(osp.join(root_path, 'smplx_optimized', 'meshes', str(frame_idx) + '.ply'))
    with open(osp.join(root_path, 'cam_params', str(frame_idx) + '.json')) as f:
        cam_param = json.load(f)

    render = render_mesh(vert.numpy(), smpl_x.face, {'focal': cam_param['focal'], 'princpt': cam_param['princpt']}, img, 1.0).astype(np.uint8)
    cv2.imwrite(osp.join(root_path, 'smplx_optimized', 'renders', str(frame_idx) + '.jpg'), render)

    frame = np.concatenate((img, render),1)
    frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(frame)
video_save.release()

