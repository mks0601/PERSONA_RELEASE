# modified from https://github.com/Tencent/MimicMotion/blob/main/inference.py
# modified from https://github.com/Tencent/MimicMotion/blob/main/mimicmotion/dwpose/preprocess.py

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
import math
import matplotlib
from pytorch3d.io import save_ply
from scipy.signal import savgol_filter
import argparse

import sys
cur_dir = osp.dirname(os.path.abspath(__file__))
root_dir = osp.join(cur_dir, '..', '..')
sys.path.insert(0, osp.join(root_dir, 'main'))
sys.path.insert(0, osp.join(root_dir, 'common'))
from utils.smpl_x import smpl_x
from utils.preprocessing import get_bbox, set_aspect_ratio, get_patch_img, get_affine_trans_mat
from utils.transforms import change_kpt_name

def draw_body_kpt(img, xy, score):
    kpt_idx_pair = [
        [1, 2],
        [1, 5],
        [2, 3],
        [3, 4],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [1, 11],
        [11, 12],
        [12, 13],
        [1, 0],
        [0, 14],
        [14, 16],
        [0, 15],
        [15, 17]
    ]

    color = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]
    
    # draw line
    stickwidth = 4
    for i, (idx1, idx2) in enumerate(kpt_idx_pair):
        if (score[idx1] < 0.3) or (score[idx2] < 0.3):
            continue
        x1, y1 = xy[idx1]
        x2, y2 = xy[idx2]
        x_mean = (x1+x2)/2.
        y_mean = (y1+y2)/2.
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
        polygon = cv2.ellipse2Poly((int(x_mean), int(y_mean)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(img, polygon, [int(c*score[idx1]*score[idx2]) for c in color[i]])
    img = (img * 0.6).astype(np.uint8)

    # draw circle
    for i in range(len(xy)):
        if score[i] < 0.3:
            continue
        cv2.circle(img, (int(xy[i,0]), int(xy[i,1])), 4, [int(c*score[i]) for c in color[i]], thickness=-1)

    return img

def draw_hand_kpt(img, xy, score):
    kpt_idx_pair = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]
    eps = 0.01

    # draw line
    for i, (idx1, idx2) in enumerate(kpt_idx_pair):
        x1, y1 = xy[idx1]
        x2, y2 = xy[idx2]
        color = matplotlib.colors.hsv_to_rgb([i / float(len(kpt_idx_pair)), 1.0, 1.0]) * int(score[idx1] * score[idx2] * 255)
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
    
    # draw circle
    for i in range(len(xy)):
        x, y = xy[i]
        if x > eps and y > eps:
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, int(score[i]*255)), thickness=-1)
    return img

def draw_face_kpt(img, xy, score):
    eps = 0.01
    for i in range(len(xy)):
        x, y = xy[i]
        if x > eps and y > eps:
            #cv2.circle(img, (int(x), int(y)), 3, [int(score[i]*255) for _ in range(3)], thickness=-1) # original
            cv2.circle(img, (int(x), int(y)), 3, [int((score[i]**2)*255) for _ in range(3)], thickness=-1) # modified
    return img

def get_tgt_motion(motion_path, tgt_shape):
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(motion_path, 'smplx_params', '*.json'))])
    smplx_params, kpts, bboxes = [], [], []
    for i, frame_idx in enumerate(frame_idx_list):
        with open(osp.join(motion_path, 'smplx_params', str(frame_idx) + '.json')) as f:
            smplx_params.append({k: torch.FloatTensor(v).cuda().view(1,-1) for k,v in json.load(f).items()})
            for k in ('jaw_pose', 'leye_pose', 'reye_pose', 'expr'):
                smplx_params[-1][k] = torch.zeros_like(smplx_params[-1][k]) # make this zero to make the generated capture easy
            smplx_params[-1]['rhand_pose'] = -smpl_x.layer.right_hand_mean.clone().cuda() # for the flat hand
            smplx_params[-1]['lhand_pose'] = -smpl_x.layer.left_hand_mean.clone().cuda() # for the flat hand
        with open(osp.join(motion_path, 'keypoints_whole_body', str(frame_idx) + '.json')) as f:
            kpts.append(np.array(json.load(f), dtype=np.float32).reshape(-1,3))
        bboxes.append(set_aspect_ratio(get_bbox(kpts[-1][:,:2], kpts[-1][:,2]>0.3), aspect_ratio=tgt_shape[1]/tgt_shape[0], extend_ratio=1))
    frame_num = len(smplx_params)

    # crop and resize kpts to tgt_shape
    kpts, bbox = np.stack(kpts).reshape(-1,3), np.stack(bboxes).mean(0)
    _, img2bb_trans, _ = get_patch_img(np.zeros((10,10,3)), bbox, tgt_shape)
    xy1 = np.concatenate((kpts[:,:2], np.ones_like(kpts[:,:1])),1)
    xy = np.dot(img2bb_trans, xy1.transpose(1,0)).transpose(1,0).reshape(frame_num,-1,2)
    kpts = kpts.reshape(frame_num,-1,3)
    kpts[:,:,:2] = xy
    return smplx_params, kpts

def get_root_trans_init(focal, princpt, kpts):
    kpts = kpts.reshape(-1,3)
    bbox = set_aspect_ratio(get_bbox(kpts, kpts[:,2]>0.2), aspect_ratio=1)
    root_trans_z = math.sqrt(focal[0]*focal[1]*2*2/bbox[2]/bbox[3]) # meter
    root_trans_x = bbox[0] + bbox[2]/2 # pixel
    root_trans_y = bbox[1] + bbox[3]/2 # pixel
    root_trans_x = (root_trans_x - princpt[0]) / focal[0] * root_trans_z # meter
    root_trans_y = (root_trans_y - princpt[1]) / focal[1] * root_trans_z # meter
    root_trans = torch.FloatTensor([root_trans_x, root_trans_y, root_trans_z]).cuda()
    return root_trans

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    assert args.motion_path, "Please set motion_path."
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
kpt_set_openpose = {
        'num': 128,
        'name': ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear', # body
            *['Face_' + str(i) for i in range(52,69)], # face contour
            *['Face_' + str(i) for i in range(1,52)], # face
            'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
            'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
        }
kpt_set_openpose['part_idx'] = {'body': range(kpt_set_openpose['name'].index('Nose'), kpt_set_openpose['name'].index('L_Ear')+1),
                            'face': range(kpt_set_openpose['name'].index('Face_52'), kpt_set_openpose['name'].index('Face_51')+1),
                            'lhand': range(kpt_set_openpose['name'].index('L_Wrist_Hand'), kpt_set_openpose['name'].index('L_Pinky_4')+1),
                            'rhand': range(kpt_set_openpose['name'].index('R_Wrist_Hand'), kpt_set_openpose['name'].index('R_Pinky_4')+1)}

# get path
args = parse_args()
root_path = args.root_path
motion_path = args.motion_path
os.makedirs(osp.join(root_path, 'cam_params'), exist_ok=True)
os.makedirs(osp.join(root_path, 'smplx_optimized', 'smplx_params'), exist_ok=True)
os.makedirs(osp.join(root_path, 'smplx_optimized', 'meshes'), exist_ok=True)

# get target motion information
h_tgt, w_tgt = 1024, 576
smplx_params, kpts = get_tgt_motion(motion_path, (h_tgt, w_tgt))
with open(osp.join(root_path, '..', 'captured', 'smplx_optimized', 'shape_param.json')) as f:
    smplx_shape = torch.FloatTensor(json.load(f)).cuda().view(1,-1)
with open(osp.join(root_path, '..', 'captured', 'smplx_optimized', 'face_offset.json')) as f:
    face_offset = torch.FloatTensor(json.load(f)).cuda().view(1,-1,3)
with open(osp.join(root_path, '..', 'captured', 'smplx_optimized', 'joint_offset.json')) as f:
    joint_offset = torch.FloatTensor(json.load(f)).cuda().view(1,-1,3)
os.system('cp ' + osp.join(motion_path, 'frame_idx_list.txt') + ' ' + osp.join(root_path, '.'))
frame_num = len(smplx_params)

# optimize global translation to make scale and translation of the projected 2D keypoints from SMPL-X similar to kpts
print('optimize root translation')
focal, princpt = (2000, 2000), (w_tgt/2, h_tgt/2)
root_trans = nn.Parameter(torch.stack([get_root_trans_init(focal, princpt, kpts[i]) for i in range(frame_num)]))
optimizer = torch.optim.Adam([root_trans], lr=0.01)
smplx_layer = copy.deepcopy(smpl_x.layer).cuda()
smplx_params_flatten = {'root_pose': [], 'body_pose': [], 'jaw_pose': [], 'leye_pose': [], 'reye_pose': [], 'lhand_pose': [], 'rhand_pose': [], 'expr': [], 'trans': []}
for k in smplx_params_flatten.keys():
    for i in range(frame_num):
        smplx_params_flatten[k].append(smplx_params[i][k])
smplx_params_flatten = {k: torch.cat(v) for k,v in smplx_params_flatten.items()}
kpt_tgt = []
for i in range(frame_num):
    kpt_tgt.append(change_kpt_name(kpts[i], kpt_set_coco['name'], smpl_x.kpt['name']))
kpt_tgt = torch.FloatTensor(np.stack(kpt_tgt)).cuda()
kpt_tgt, kpt_valid = kpt_tgt[:,:,:2], (kpt_tgt[:,:,2:] > 0.2).float()
kpt_valid[:,[i for i in range(smpl_x.kpt['num']) if smpl_x.kpt['name'][i] in ['L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel']],:] *= 10
for itr in range(500):
    if itr == 100:
        for g in optimizer.param_groups:
            g['lr'] = 0.001
    optimizer.zero_grad()
    output = smplx_layer(global_orient=smplx_params_flatten['root_pose'],
                        body_pose=smplx_params_flatten['body_pose'],
                        jaw_pose=smplx_params_flatten['jaw_pose'],
                        leye_pose=smplx_params_flatten['leye_pose'],
                        reye_pose=smplx_params_flatten['reye_pose'],
                        left_hand_pose=smplx_params_flatten['lhand_pose'],
                        right_hand_pose=smplx_params_flatten['rhand_pose'],
                        expression=smplx_params_flatten['expr'],
                        betas=smplx_shape,
                        face_offset=face_offset,
                        joint_offset=joint_offset)
    kpt_cam = output.joints[:,smpl_x.kpt['idx'],:]
    kpt_cam = kpt_cam - kpt_cam[:,smpl_x.kpt['root_idx'],None,:] + root_trans.view(-1,1,3)
    x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * focal[0] + princpt[0]
    y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * focal[1] + princpt[1]
    kpt_proj = torch.stack((x,y),2)
    loss = (torch.abs(kpt_proj - kpt_tgt) * kpt_valid).mean()
    loss.backward()
    optimizer.step()
    if itr in list(range(0,500,10)):
        print('itr: ' + str(itr) + ', loss: ' + str(float(loss.detach())))

# smooth optimized root translation
root_trans = root_trans.detach().cpu().numpy()
root_trans = savgol_filter(root_trans, window_length=9, polyorder=2, axis=0)
root_trans = torch.FloatTensor(root_trans).cuda()

# get the final keypoints, used to make the pose video (input of MimicMotion) and save data
print('save camera and smplx parameters')
kpt_tgt, kpt_vis = [], []
for i in range(frame_num):
    # get projected 2D keypoints
    output = smplx_layer(global_orient=smplx_params[i]['root_pose'],
                        body_pose=smplx_params[i]['body_pose'],
                        jaw_pose=smplx_params[i]['jaw_pose'],
                        leye_pose=smplx_params[i]['leye_pose'],
                        reye_pose=smplx_params[i]['reye_pose'],
                        left_hand_pose=smplx_params[i]['lhand_pose'],
                        right_hand_pose=smplx_params[i]['rhand_pose'],
                        expression=smplx_params[i]['expr'],
                        betas=smplx_shape,
                        face_offset=face_offset,
                        joint_offset=joint_offset)
    kpt_cam = output.joints[:,smpl_x.kpt['idx'],:]
    vert_cam = output.vertices
    trans = -kpt_cam[:,smpl_x.kpt['root_idx'],None,:] + root_trans[i].view(1,1,3)
    kpt_cam = kpt_cam + trans
    vert_cam = vert_cam + trans
    x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * focal[0] + princpt[0]
    y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * focal[1] + princpt[1]
    kpt_proj = torch.stack((x,y,torch.ones_like(x)),2).detach().cpu().numpy()[0]
    kpt_proj_save = change_kpt_name(kpt_proj, smpl_x.kpt['name'], kpt_set_coco['name'])
    kpt_proj_save[:,2] *= kpts[i][:,2] # score
    kpt_proj = change_kpt_name(kpt_proj, smpl_x.kpt['name'], kpt_set_openpose['name'])[:,:2]
    kpt_tgt.append(kpt_proj_save)

    # fill missing keypoints
    kpt_proj[kpt_set_openpose['name'].index('L_Wrist_Hand')] = kpt_proj[kpt_set_openpose['name'].index('L_Elbow')] + (kpt_proj[kpt_set_openpose['name'].index('L_Wrist')] - kpt_proj[kpt_set_openpose['name'].index('L_Elbow')]) * 1.1
    kpt_proj[kpt_set_openpose['name'].index('R_Wrist_Hand')] = kpt_proj[kpt_set_openpose['name'].index('R_Elbow')] + (kpt_proj[kpt_set_openpose['name'].index('R_Wrist')] - kpt_proj[kpt_set_openpose['name'].index('R_Elbow')]) * 1.1
    kpt_proj[kpt_set_openpose['name'].index('Neck')] = (kpt_proj[kpt_set_openpose['name'].index('R_Shoulder')] + kpt_proj[kpt_set_openpose['name'].index('L_Shoulder')])/2
    kpt_score = change_kpt_name(kpts[i][:,2], kpt_set_coco['name'], kpt_set_openpose['name'])
    kpt_score[kpt_set_openpose['name'].index('Neck')] = (kpt_score[kpt_set_openpose['name'].index('R_Shoulder')]>0.3) * (kpt_score[kpt_set_openpose['name'].index('L_Shoulder')]>0.3)
    kpt_vis.append({'xy': kpt_proj, 'score': kpt_score})
    
    # save camera parameter
    with open(osp.join(root_path, 'cam_params', str(i) + '.json'), 'w') as f:
        json.dump({'R': np.eye(3).astype(np.float32).tolist(), 't': np.zeros((3), dtype=np.float32).tolist(), 'focal': focal, 'princpt': princpt}, f)

    # save smplx parameters/meshes with optimized translation
    smplx_params[i]['trans'] = trans.view(3)
    with open(osp.join(root_path, 'smplx_optimized', 'smplx_params', str(i) + '.json'), 'w') as f:
        json.dump({'root_pose': smplx_params[i]['root_pose'].detach().cpu().numpy().reshape(3).tolist(), \
                'body_pose': smplx_params[i]['body_pose'].detach().cpu().numpy().reshape(-1,3).tolist(), \
                'jaw_pose': smplx_params[i]['jaw_pose'].detach().cpu().numpy().reshape(3).tolist(), \
                'leye_pose': smplx_params[i]['leye_pose'].detach().cpu().numpy().reshape(3).tolist(), \
                'reye_pose': smplx_params[i]['reye_pose'].detach().cpu().numpy().reshape(3).tolist(), \
                'lhand_pose': smplx_params[i]['lhand_pose'].detach().cpu().numpy().reshape(-1,3).tolist(), \
                'rhand_pose': smplx_params[i]['rhand_pose'].detach().cpu().numpy().reshape(-1,3).tolist(), \
                'expr': smplx_params[i]['expr'].detach().cpu().numpy().reshape(-1).tolist(), \
                'trans': smplx_params[i]['trans'].detach().cpu().numpy().reshape(-1).tolist()}, f)
    save_ply(osp.join(root_path, 'smplx_optimized', 'meshes', str(i) + '.ply'), vert_cam[0].detach().cpu(), torch.LongTensor(smpl_x.face))

# save image to animate (input of MimicMotion) after transforming it to kpt_tgt
print('save image to animate')
img_src = cv2.imread(osp.join(root_path, '..', 'captured', 'images', '0.png'))
mask_src = cv2.imread(osp.join(root_path, '..', 'captured', 'masks', '0.png'))
img_src = (img_src * (mask_src>0) + 255*(mask_src==0)).astype(np.uint8) # white bg
h_src, w_src = img_src.shape[:2]
kpt_names = ['Nose', 'R_Shoulder', 'L_Shoulder', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear']
kpt_idxs = [kpt_set_coco['name'].index(name) for name in kpt_names]
kpt_tgt = np.stack(kpt_tgt)[:,kpt_idxs,]
kpt_tgt, kpt_tgt_valid = kpt_tgt[:,:,:2], kpt_tgt[:,:,2] > 0.3
with open(osp.join(root_path, '..', 'captured', 'keypoints_whole_body', '0.json')) as f:
    kpt_src = np.array(json.load(f), dtype=np.float32)[kpt_idxs,:]
    kpt_src, kpt_src_valid = kpt_src[:,:2], kpt_src[:,2] > 0.3
    kpt_src = kpt_src[kpt_src_valid,:]
kpt_tgt, kpt_tgt_valid = kpt_tgt[:,kpt_src_valid,:], kpt_tgt_valid[:,kpt_src_valid]
kpt_tgt = kpt_tgt[np.all(kpt_tgt_valid,1),:,:]
scale, trans_y = np.polyfit(np.tile(kpt_src[None,:,1],(kpt_tgt.shape[0],1)).reshape(-1), kpt_tgt[:,:,1].reshape(-1), 1)
trans_x = (kpt_tgt[:,:,0] - kpt_src[None,:,0]*scale).mean()
img_src = cv2.resize(img_src, (int(w_src*scale), int(h_src*scale)))
h_src, w_src = img_src.shape[:2]
xmin = int(trans_x) 
ymin = int(trans_y) 
xmax, ymax = xmin+w_src, ymin+h_src
img_save = img_src[max(-ymin,0):min(h_src-(ymax-h_tgt),h_src), max(-xmin,0):min(w_src-(xmax-w_tgt),w_src),:]
img_save = np.pad(img_save, ((max(ymin,0), h_tgt-min(ymin+h_src,h_tgt)), (max(xmin,0),w_tgt-min(xmin+w_src,w_tgt)), (0,0)), constant_values=255)
assert img_save.shape[:2] == (h_tgt, w_tgt)
cv2.imwrite(osp.join(root_path, 'img_to_anim.png'), img_save)

# save the pose video (input of MimicMotion)
print('save video to animate')
video_save = cv2.VideoWriter(osp.join(root_path, 'video_to_anim.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w_tgt, h_tgt))
for i in tqdm(range(frame_num)):
    kpt_proj, kpt_score = kpt_vis[i]['xy'], kpt_vis[i]['score']
    
    # visualize keypoints (following MimicMotion's preprocessing stage)
    sz = min(h_tgt, w_tgt)
    sr = 2160 / sz
    vis = np.zeros((int(h_tgt*sr), int(w_tgt*sr), 3), dtype=np.uint8)
    kpt_proj *= sr
    vis = draw_body_kpt(vis, kpt_proj[kpt_set_openpose['part_idx']['body']], kpt_score[kpt_set_openpose['part_idx']['body']])
    vis = draw_hand_kpt(vis, kpt_proj[kpt_set_openpose['part_idx']['lhand']], kpt_score[kpt_set_openpose['part_idx']['lhand']])
    vis = draw_hand_kpt(vis, kpt_proj[kpt_set_openpose['part_idx']['rhand']], kpt_score[kpt_set_openpose['part_idx']['rhand']])
    vis = draw_face_kpt(vis, kpt_proj[kpt_set_openpose['part_idx']['face']], kpt_score[kpt_set_openpose['part_idx']['face']])
    vis = cv2.cvtColor(cv2.resize(vis, (w_tgt, h_tgt)), cv2.COLOR_BGR2RGB)
    video_save.write(vis)
video_save.release()

