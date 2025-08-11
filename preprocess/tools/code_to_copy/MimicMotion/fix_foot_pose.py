import numpy as np
import cv2
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import torch
import torch.optim
import torch.nn as nn
import copy
from pytorch3d.io import save_ply
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle
from scipy.signal import savgol_filter
import argparse

import sys
cur_dir = osp.dirname(os.path.abspath(__file__))
root_dir = osp.join(cur_dir, '..', '..', '..')
sys.path.insert(0, osp.join(root_dir, 'main'))
sys.path.insert(0, osp.join(root_dir, 'common'))
from utils.smpl_x import smpl_x
from utils.transforms import change_kpt_name

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
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# keypoint set
kpt_set_coco = {
        'num': 133,
        'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
            *['Face_' + str(i) for i in range(52,69)], # face contour
            *['Face_' + str(i) for i in range(1,52)], # face
            'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
            'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
        }

# get path
args = parse_args()
root_path = args.root_path
frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(root_path, 'smplx_optimized', 'smplx_params', '*.json'))])
frame_num = len(frame_idx_list)

# load smplx parameters, keypoints, and camera parameters
smplx_params = {'root_pose': [], 'body_pose': [], 'jaw_pose': [], 'leye_pose': [], 'reye_pose': [], 'lhand_pose': [], 'rhand_pose': [], 'expr': [], 'trans': []}
kpts, cam_params = [], {'R': [], 't': [], 'focal': [], 'princpt': []}
for frame_idx in frame_idx_list:
    with open(osp.join(root_path, 'smplx_optimized', 'smplx_params', str(frame_idx) + '.json')) as f:
        for k,v in json.load(f).items():
            smplx_params[k].append(torch.FloatTensor(v))
    with open(osp.join(root_path, 'keypoints_whole_body', str(frame_idx) + '.json')) as f:
        kpts.append(np.array(json.load(f), dtype=np.float32).reshape(-1,3))
    with open(osp.join(root_path, 'cam_params', str(frame_idx) + '.json')) as f:
        for k,v in json.load(f).items():
            cam_params[k].append(torch.FloatTensor(v))
smplx_params = {k: torch.stack(v).cuda() for k,v in smplx_params.items()}
kpts = np.stack(kpts)
cam_params = {k: torch.stack(v).cuda() for k,v in cam_params.items()}

# load ID information
with open(osp.join(root_path, '..', 'captured', 'smplx_optimized', 'shape_param.json')) as f:
    smplx_shape = torch.FloatTensor(json.load(f)).cuda().view(1,-1)
with open(osp.join(root_path, '..', 'captured', 'smplx_optimized', 'face_offset.json')) as f:
    face_offset = torch.FloatTensor(json.load(f)).cuda().view(1,-1,3)
with open(osp.join(root_path, '..', 'captured', 'smplx_optimized', 'joint_offset.json')) as f:
    joint_offset = torch.FloatTensor(json.load(f)).cuda().view(1,-1,3)

# optimize foot poses
print('optimize foot pose')
rankle_poses = nn.Parameter(smplx_params['body_pose'][:,smpl_x.joint['name'].index('R_Ankle')-1,:]) # body_pose does not have root joint
lankle_poses = nn.Parameter(smplx_params['body_pose'][:,smpl_x.joint['name'].index('L_Ankle')-1,:]) # body_pose does not have root joint
optimizer = torch.optim.Adam([rankle_poses,lankle_poses], lr=0.01)
smplx_layer = copy.deepcopy(smpl_x.layer).cuda()
kpt_tgt = []
for i in range(frame_num):
    kpt_tgt.append(change_kpt_name(kpts[i], kpt_set_coco['name'], smpl_x.kpt['name']))
kpt_tgt = torch.FloatTensor(np.stack(kpt_tgt)).cuda()
kpt_tgt, kpt_valid = kpt_tgt[:,:,:2], (kpt_tgt[:,:,2:] > 0.3).float()
foot_kpt_idxs = [i for i in range(smpl_x.kpt['num']) if smpl_x.kpt['name'][i] in ['L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel']]
for itr in range(500):
    if itr == 100:
        for g in optimizer.param_groups:
            g['lr'] = 0.001
    optimizer.zero_grad()

    body_pose = smplx_params['body_pose']
    body_pose = torch.cat((body_pose[:,:smpl_x.joint['name'].index('R_Ankle')-1,:], rankle_poses[:,None,:], body_pose[:,smpl_x.joint['name'].index('R_Ankle'):,:]),1)
    body_pose = torch.cat((body_pose[:,:smpl_x.joint['name'].index('L_Ankle')-1,:], lankle_poses[:,None,:], body_pose[:,smpl_x.joint['name'].index('L_Ankle'):,:]),1)
    output = smplx_layer(global_orient=smplx_params['root_pose'],
                        body_pose=body_pose,
                        jaw_pose=smplx_params['jaw_pose'],
                        leye_pose=smplx_params['leye_pose'],
                        reye_pose=smplx_params['reye_pose'],
                        left_hand_pose=smplx_params['lhand_pose'],
                        right_hand_pose=smplx_params['rhand_pose'],
                        expression=smplx_params['expr'],
                        transl=smplx_params['trans'],
                        betas=smplx_shape,
                        face_offset=face_offset,
                        joint_offset=joint_offset)
    kpt_cam = output.joints[:,smpl_x.kpt['idx'],:]
    x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * cam_params['focal'][:,None,0] + cam_params['princpt'][:,None,0]
    y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * cam_params['focal'][:,None,1] + cam_params['princpt'][:,None,1]
    kpt_proj = torch.stack((x,y),2)
    loss = (torch.abs(kpt_proj - kpt_tgt) * kpt_valid)[:,foot_kpt_idxs,:].mean()
    loss.backward()
    optimizer.step()
    if itr in list(range(0,500,10)):
        print('itr: ' + str(itr) + ', loss: ' + str(float(loss.detach())))
smplx_params['body_pose'] = body_pose

# smooth optimized foot poses
rankle_poses = smplx_params['body_pose'][:,smpl_x.joint['name'].index('R_Ankle')-1,:].detach().cpu().numpy() # body_pose does not have root joint
lankle_poses = smplx_params['body_pose'][:,smpl_x.joint['name'].index('L_Ankle')-1,:].detach().cpu().numpy() # body_pose does not have root joint
ankle_poses = np.concatenate((rankle_poses, lankle_poses),1).reshape(frame_num,-1)
ankle_poses = smoothen_poses(ankle_poses, 9).reshape(frame_num,2,3)
rankle_poses, lankle_poses = torch.FloatTensor(ankle_poses[:,0,:]).cuda(), torch.FloatTensor(ankle_poses[:,1,:]).cuda()
smplx_params['body_pose'][:,smpl_x.joint['name'].index('R_Ankle')-1,:] = rankle_poses
smplx_params['body_pose'][:,smpl_x.joint['name'].index('L_Ankle')-1,:] = lankle_poses

# save optimized smplx parameters
print('save smplx parameters')
for i in range(frame_num):
    frame_idx = frame_idx_list[i]
    output = smplx_layer(global_orient=smplx_params['root_pose'][i].view(1,-1),
                        body_pose=smplx_params['body_pose'][i].view(1,-1),
                        jaw_pose=smplx_params['jaw_pose'][i].view(1,-1),
                        leye_pose=smplx_params['leye_pose'][i].view(1,-1),
                        reye_pose=smplx_params['reye_pose'][i].view(1,-1),
                        left_hand_pose=smplx_params['lhand_pose'][i].view(1,-1),
                        right_hand_pose=smplx_params['rhand_pose'][i].view(1,-1),
                        expression=smplx_params['expr'][i].view(1,-1),
                        transl=smplx_params['trans'][i].view(1,-1),
                        betas=smplx_shape,
                        face_offset=face_offset,
                        joint_offset=joint_offset)
    vert_cam = output.vertices
    
    # save smplx parameters/meshes with optimized translation
    with open(osp.join(root_path, 'smplx_optimized', 'smplx_params', str(frame_idx) + '.json'), 'w') as f:
        json.dump({'root_pose': smplx_params['root_pose'][i].detach().cpu().numpy().reshape(3).tolist(), \
                'body_pose': smplx_params['body_pose'][i].detach().cpu().numpy().reshape(-1,3).tolist(), \
                'jaw_pose': smplx_params['jaw_pose'][i].detach().cpu().numpy().reshape(3).tolist(), \
                'leye_pose': smplx_params['leye_pose'][i].detach().cpu().numpy().reshape(3).tolist(), \
                'reye_pose': smplx_params['reye_pose'][i].detach().cpu().numpy().reshape(3).tolist(), \
                'lhand_pose': smplx_params['lhand_pose'][i].detach().cpu().numpy().reshape(-1,3).tolist(), \
                'rhand_pose': smplx_params['rhand_pose'][i].detach().cpu().numpy().reshape(-1,3).tolist(), \
                'expr': smplx_params['expr'][i].detach().cpu().numpy().reshape(-1).tolist(), \
                'trans': smplx_params['trans'][i].detach().cpu().numpy().reshape(-1).tolist()}, f)

    save_ply(osp.join(root_path, 'smplx_optimized', 'meshes', str(frame_idx) + '.ply'), vert_cam[0].detach().cpu(), torch.LongTensor(smpl_x.face))

