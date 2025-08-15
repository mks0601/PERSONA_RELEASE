import argparse
import numpy as np
import cv2
from config import cfg
import torch
import json
import os
import os.path as osp
from utils.smpl_x import smpl_x
from utils.flame import flame
from glob import glob
from tqdm import tqdm
from pytorch3d.io import load_ply
from utils.vis import render_mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--split', type=str, dest='split')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    assert args.split in ['captured', 'test']
    return args

def main():
    args = parse_args()
    cfg.set_args(args.subject_id, split=args.split)
    
    root_path = osp.join('..', 'data', 'subjects', cfg.subject_id, cfg.split)
    save_root_path = osp.join(root_path, 'smplx_optimized')
    os.makedirs(osp.join(save_root_path, 'renders'), exist_ok=True)
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(root_path, 'images', '*.png'))])

    for frame_idx in tqdm(frame_idx_list):
        
        # load image
        img_path = osp.join(root_path, 'images', str(frame_idx) + '.png')
        img = cv2.imread(img_path)
        
        # load camera parameter
        cam_param_path = osp.join(root_path, 'cam_params', str(frame_idx) + '.json')
        with open(cam_param_path) as f:
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
        
        # render smplx mesh
        mesh_path = osp.join(save_root_path, 'meshes', str(frame_idx) + '.ply')
        vert, face = load_ply(mesh_path)
        render = render_mesh(vert.numpy(), smpl_x.face, {'focal': cam_param['focal'], 'princpt': cam_param['princpt']}, img, 1.0)
        cv2.imwrite(osp.join(save_root_path, 'renders', str(frame_idx) + '.jpg'), render)

        # render flame mesh
        if args.split == 'captured':
            mesh_path = osp.join(save_root_path, 'meshes', str(frame_idx) + '_flame.ply')
            vert, face = load_ply(mesh_path)
            render = render_mesh(vert.numpy(), flame.face, {'focal': cam_param['focal'], 'princpt': cam_param['princpt']}, img, 1.0)
            cv2.imwrite(osp.join(save_root_path, 'renders', str(frame_idx) + '_flame.jpg'), render)

if __name__ == "__main__":
    main()
