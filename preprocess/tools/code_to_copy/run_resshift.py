import os
import os.path as osp
from glob import glob
import cv2
import argparse
import json
import numpy as np
from tqdm import tqdm
from pytorch3d.io import load_ply
import math
import torch
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

def get_bbox(kpt_img, kpt_valid, extend_ratio=1.2):

    x_img, y_img = kpt_img[:,0], kpt_img[:,1]
    x_img = x_img[kpt_valid==1]; y_img = y_img[kpt_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def set_aspect_ratio(bbox, extend_ratio=1.25):
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 1
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*extend_ratio
    bbox[3] = h*extend_ratio
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def get_patch_img(cvimg, bbox, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = get_affine_trans_mat(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0])
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = get_affine_trans_mat(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], inv=True)
    return img_patch, trans, inv_trans

def get_affine_trans_mat(c_x, c_y, src_w, src_h, dst_w, dst_h, inv=False):
    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = np.array([0, src_h * 0.5], dtype=np.float32)
    src_rightdir = np.array([src_w * 0.5, 0], dtype=np.float32)

    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    trans = trans.astype(np.float32)
    return trans

kpt_info = {
        'num': 133,
        'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
            *['Face_' + str(i) for i in range(52,69)], # face contour
            *['Face_' + str(i) for i in range(1,52)], # face
            'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
            'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
        }
kpt_info['part_idx'] = {'face': [i for i in range(kpt_info['num']) if 'Face' in kpt_info['name'][i]]}

args = parse_args()
root_path = args.root_path
patch_shape = (64, 64)
patch_save_path = osp.join(root_path, 'face_patches')
os.makedirs(patch_save_path, exist_ok=True)
os.system('rm -rf ' + osp.join(patch_save_path, '*'))

os.makedirs(osp.join(patch_save_path, 'images_orig'), exist_ok=True)
os.makedirs(osp.join(patch_save_path, 'images'), exist_ok=True)
os.makedirs(osp.join(patch_save_path, 'bbox'), exist_ok=True)

# crop face and save face images
frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(root_path, 'keypoints_whole_body', '*.json'))])
for frame_idx in tqdm(frame_idx_list):
    
    # load keypoints
    with open(osp.join(root_path, 'keypoints_whole_body', str(frame_idx) + '.json')) as f:
        kpt = np.array(json.load(f), dtype=np.float32).reshape(-1,3)
    face_kpt = kpt[kpt_info['part_idx']['face'],:]
    
    # face bbox
    face_bbox = get_bbox(face_kpt[:,:2], face_kpt[:,2]>0.2)
    face_bbox[1] = face_bbox[1] - face_bbox[3]*0.2
    face_bbox = set_aspect_ratio(face_bbox)

    # crop face patch and save
    img_path = osp.join(root_path, 'images', str(frame_idx) + '.png')
    img = cv2.imread(img_path)
    face_patch, _, _ = get_patch_img(img, face_bbox, patch_shape)
    cv2.imwrite(osp.join(patch_save_path, 'images_orig', str(frame_idx) + '.png'), face_patch)
    with open(osp.join(patch_save_path, 'bbox', str(frame_idx) + '.json'), 'w') as f:
        json.dump(face_bbox.tolist(), f)


# run ResShift and save output
sr_output_path = './face_output_sr'
os.system('rm -rf ' + osp.join(sr_output_path, '*'))
cmd = 'python inference_resshift.py -i ' + osp.join(patch_save_path, 'images_orig') + ' -o ' + sr_output_path + ' --task realsr --scale 4 --version v3'
os.system(cmd)
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(patch_save_path, 'images_orig', '*.png'))])
for frame_idx in tqdm(frame_idx_list):
    cmd = 'mv ' + osp.join(sr_output_path, str(frame_idx) + '.png') + ' ' + osp.join(patch_save_path, 'images', str(frame_idx) + '.png')
    os.system(cmd)

