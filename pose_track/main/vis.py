import argparse
import numpy as np
import cv2
from config import cfg
import torch
import json
import os
import os.path as osp
from utils.smpl_x import smpl_x
from glob import glob
from tqdm import tqdm
from pytorch3d.io import load_ply
from utils.vis import render_mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--vis_format', type=str, dest='vis_format')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    assert args.vis_format in ['image', 'video'], "Please set vis_format among 'image' and 'video'"
    return args

def main():
    args = parse_args()
    cfg.set_args(args.subject_id)
    vis_format = args.vis_format
    
    root_path = osp.join('..', 'data', 'subjects', cfg.subject_id)
    save_root_path = osp.join(root_path, 'smplx_optimized')
    os.makedirs(osp.join(save_root_path, 'renders'), exist_ok=True)
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(save_root_path, 'meshes', '*.ply'))])

    if vis_format == 'video':
        img_height, img_width = cv2.imread(osp.join(root_path, 'images', str(frame_idx_list[0]) + '.png')).shape[:2]
        video_save = cv2.VideoWriter(osp.join(root_path, 'smplx_optimized.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))

    for frame_idx in tqdm(frame_idx_list):
        
        # load image
        img_path = osp.join(root_path, 'images', str(frame_idx) + '.png')
        img = cv2.imread(img_path)
        
        # load camera parameter
        cam_param_path = osp.join(root_path, 'cam_params', str(frame_idx) + '.json')
        with open(cam_param_path) as f:
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
        
        # load mesh
        mesh_path = osp.join(save_root_path, 'meshes', str(frame_idx) + '.ply')
        vert, face = load_ply(mesh_path)

        # render
        render = render_mesh(vert.numpy(), smpl_x.face, {'focal': cam_param['focal'], 'princpt': cam_param['princpt']}, img, 1.0)
        cv2.imwrite(osp.join(save_root_path, 'renders', str(frame_idx) + '.jpg'), render)
        if vis_format == 'video':
            frame = np.concatenate((img, render),1)
            frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
            video_save.write(frame.astype(np.uint8))

if __name__ == "__main__":
    main()
