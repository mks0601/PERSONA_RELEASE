import torch
import numpy as np
import os
import os.path as osp
from scipy.signal import savgol_filter
import json
from glob import glob
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle
import cv2
os.environ["PYOPENGL_PLATFORM"] = "egl"
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRenderer,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)
import argparse
import sys
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'common'))

def fix_quaternions(quats):
    """
    From https://github.com/facebookresearch/QuaterNet/blob/ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py

    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    :param quats: A numpy array of shape (F, N, 4).
    :return: A numpy array of the same shape.
    """
    assert len(quats.shape) == 3
    assert quats.shape[-1] == 4

    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result

def smoothen_poses(poses, window_length):
    """Smooth joint angles. Poses and global_root_orient should be given as rotation vectors."""
    n_joints = poses.shape[1] // 3

    # Convert poses to quaternions.
    qs = matrix_to_quaternion(axis_angle_to_matrix(torch.FloatTensor(poses).view(-1,3))).numpy()
    qs = qs.reshape((-1, n_joints, 4))
    qs = fix_quaternions(qs)

    # Smooth the quaternions.
    qs_smooth = []
    for j in range(n_joints):
        qss = savgol_filter(qs[:, j], window_length=window_length, polyorder=2, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / np.linalg.norm(qs_clean, axis=-1, keepdims=True)

    ps_clean = matrix_to_axis_angle(quaternion_to_matrix(torch.FloatTensor(qs_clean).view(-1,4))).numpy()
    ps_clean = np.reshape(ps_clean, [-1, n_joints * 3])
    return ps_clean

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--smooth_length', type=int, default=9, dest='smooth_length')
    args = parser.parse_args()
    return args

# get paths
args = parse_args()
root_path = args.root_path
smplx_param_path_list = glob(osp.join(root_path, 'smplx', 'params', '*.json'))
frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in smplx_param_path_list])
frame_num = len(frame_idx_list)

# load smplx parameters of all frames
smplx_params = {}
for frame_idx in frame_idx_list:
    smplx_param_path = osp.join(root_path, 'smplx', 'params', str(frame_idx) + '.json')
    with open(smplx_param_path) as f:
        smplx_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
    smplx_params[frame_idx] = smplx_param
    keys = smplx_param.keys()

# smooth smplx parameters
for key in keys:
    if 'pose' in key:
        pose = np.stack([smplx_params[frame_idx][key].reshape(-1) for frame_idx in frame_idx_list])
        pose = smoothen_poses(pose, window_length=args.smooth_length)
        for i, frame_idx in enumerate(frame_idx_list):
            smplx_params[frame_idx][key] = pose[i]
            if key in ['body_pose', 'lhand_pose', 'rhand_pose']:
                smplx_params[frame_idx][key] = smplx_params[frame_idx][key].reshape(-1,3)
    else:
        item = np.stack([smplx_params[frame_idx][key] for frame_idx in frame_idx_list])
        item = savgol_filter(item, window_length=args.smooth_length, polyorder=2, axis=0)
        for i, frame_idx in enumerate(frame_idx_list):
            smplx_params[frame_idx][key] = item[i]

# save smoothed smplx parameters
for frame_idx in tqdm(frame_idx_list):
    smplx_param = smplx_params[frame_idx]
    
    # load and replace original smplx parameter
    with open(osp.join(root_path, 'smplx', 'params', str(frame_idx) + '.json')) as f:
        smplx_param_save = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
    smplx_param_save['lhand_pose'] = smplx_param['lhand_pose']
    smplx_param_save['rhand_pose'] = smplx_param['rhand_pose']
    lwrist_idx, rwrist_idx = 19, 20
    body_pose = smplx_param_save['body_pose'].reshape(-1,3)
    body_pose[lwrist_idx] = smplx_param['body_pose'][lwrist_idx]
    body_pose[rwrist_idx] = smplx_param['body_pose'][rwrist_idx]
    smplx_param_save['body_pose'] = body_pose

    # save smplx parameter
    with open(osp.join(root_path, 'smplx', 'params', str(frame_idx) + '.json'), 'w') as f:
        json.dump({k: v.tolist() for k,v in smplx_param_save.items()}, f)
    

