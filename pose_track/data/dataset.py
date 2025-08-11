import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from config import cfg
from utils.smpl_x import smpl_x
from utils.preprocessing import load_img, get_bbox, set_aspect_ratio, get_patch_img
from utils.transforms import change_kpt_name
import json
import math

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.root_path = osp.join('..', 'data', 'subjects', cfg.subject_id)
        self.transform = transform
        self.kpt = {
                    'num': 133,
                    'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
                        *['Face_' + str(i) for i in range(52,69)], # face contour
                        *['Face_' + str(i) for i in range(1,52)], # face
                        'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
                        'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
                    }
        self.cam_params, self.kpts, self.depth_paths, self.smplx_params, self.frame_idx_list = self.load_data()
        self.get_smplx_trans_init() # get initial smplx translation 

    def load_data(self):
        # load camera parameter
        cam_params = {}
        cam_param_path_list = glob(osp.join(self.root_path, 'cam_params', '*.json'))
        for cam_param_path in cam_param_path_list:
            frame_idx = int(cam_param_path.split('/')[-1][:-5])
            with open(cam_param_path) as f:
                cam_params[frame_idx] = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}

        # load keypoints
        kpts = {}
        kpt_path_list = glob(osp.join(self.root_path, 'keypoints_whole_body', '*.json'))
        for kpt_path in kpt_path_list:
            frame_idx = int(kpt_path.split('/')[-1][:-5])
            with open(kpt_path) as f:
                kpt = np.array(json.load(f), dtype=np.float32)
            kpt = change_kpt_name(kpt, self.kpt['name'], smpl_x.kpt['name'])
            kpts[frame_idx] = kpt

        # depth map paths
        depth_paths = {}
        depth_path_list = glob(osp.join(self.root_path, 'depths', '*.npy'))
        for depth_path in depth_path_list:
            if '_mask.npy' in depth_path:
                continue
            frame_idx = int(depth_path.split('/')[-1][:-4])
            depth_paths[frame_idx] = depth_path

        # load initial smplx parameters
        smplx_params = {}
        smplx_param_path_list = glob(osp.join(self.root_path, 'smplx_init', '*.json'))
        for smplx_param_path in smplx_param_path_list:
            frame_idx = int(smplx_param_path.split('/')[-1][:-5])
            with open(smplx_param_path) as f:
                smplx_params[frame_idx] = {k: torch.FloatTensor(v) for k,v in json.load(f).items()}
            if len(smplx_params[frame_idx]['expr']) < smpl_x.expr_param_dim:
                expr = torch.zeros((smpl_x.expr_param_dim)).float()
                expr[:len(smplx_params[frame_idx]['expr'])] = smplx_params[frame_idx]['expr']
                smplx_params[frame_idx]['expr'] = expr
            elif len(smplx_params[frame_idx]['expr']) > smpl_x.expr_param_dim:
                smplx_params[frame_idx]['expr'] = smplx_params[frame_idx]['expr'][:smpl_x.expr_param_dim]

        frame_idx_list = [int(x) for x in cam_params.keys()]
        return cam_params, kpts, depth_paths, smplx_params, frame_idx_list

    def get_smplx_trans_init(self):
        for i in range(len(self.frame_idx_list)):
            frame_idx = self.frame_idx_list[i]
            focal, princpt = self.cam_params[frame_idx]['focal'], self.cam_params[frame_idx]['princpt'].copy()

            kpt = self.kpts[frame_idx]
            kpt_img = kpt[:,:2]
            kpt_valid = (kpt[:,2:] > 0.5).astype(np.float32)
            bbox = get_bbox(kpt_img, kpt_valid[:,0])
            bbox = set_aspect_ratio(bbox, aspect_ratio=cfg.kpt_proj_shape[1]/cfg.kpt_proj_shape[0])
            
            # get the root translation in the camera coordinate system
            t_z = math.sqrt(focal[0]*focal[1]*cfg.body_3d_size*cfg.body_3d_size/(bbox[2]*bbox[3])) # meter
            t_x = bbox[0] + bbox[2]/2 # pixel
            t_y = bbox[1] + bbox[3]/2 # pixel
            t_x = (t_x - princpt[0]) / focal[0] * t_z # meter
            t_y = (t_y - princpt[1]) / focal[1] * t_z # meter
            t_xyz = torch.FloatTensor([t_x, t_y, t_z]) 
            self.smplx_params[frame_idx]['trans'] = t_xyz

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        frame_idx = self.frame_idx_list[idx]

        # 2D keypoint
        kpt_img = self.kpts[frame_idx][:,:2]
        kpt_valid = (self.kpts[frame_idx][:,2:] > 0.3).astype(np.float32)
        bbox = get_bbox(kpt_img, kpt_valid[:,0])
        bbox = set_aspect_ratio(bbox, aspect_ratio=cfg.kpt_proj_shape[1]/cfg.kpt_proj_shape[0])

        # keypoint affine transformation
        _, img2bb_trans, bb2img_trans = get_patch_img(np.ones((10,10,3),dtype=np.uint8), bbox, cfg.kpt_proj_shape) # get transformation matrix with dummy image
        kpt_img_xy1 = np.concatenate((kpt_img, np.ones_like(kpt_img[:,:1])),1)
        kpt_img = np.dot(img2bb_trans, kpt_img_xy1.transpose(1,0)).transpose(1,0)

        # load depth
        _, img2bb_trans, bb2img_trans = get_patch_img(np.ones((10,10,3),dtype=np.uint8), bbox, cfg.depth_render_shape) # get transformation matrix with dummy image
        depth = np.load(self.depth_paths[frame_idx])
        depth_mask = np.load(self.depth_paths[frame_idx][:-4] + '_mask.npy').astype(np.float32)
        depth = cv2.warpAffine(depth, img2bb_trans, (cfg.depth_render_shape[1], cfg.depth_render_shape[0]), flags=cv2.INTER_LINEAR)
        depth_mask = cv2.warpAffine(depth_mask, img2bb_trans, (cfg.depth_render_shape[1], cfg.depth_render_shape[0]), flags=cv2.INTER_LINEAR)
        depth = (depth * (depth_mask==1))[None,:,:] 

        # smplx parameter
        smplx_param = self.smplx_params[frame_idx]

        # modify intrincis to directly project 3D coordinates to cfg.kpt_proj_shape space
        focal = self.cam_params[frame_idx]['focal']
        princpt = self.cam_params[frame_idx]['princpt']
        focal_kpt = np.array([focal[0] / bbox[2] * cfg.kpt_proj_shape[1], focal[1] / bbox[3] * cfg.kpt_proj_shape[0]], dtype=np.float32)
        princpt_kpt = np.array([(princpt[0] - bbox[0]) / bbox[2] * cfg.kpt_proj_shape[1], (princpt[1] - bbox[1]) / bbox[3] * cfg.kpt_proj_shape[1]], dtype=np.float32)
        focal_depth = np.array([focal[0] / bbox[2] * cfg.depth_render_shape[1], focal[1] / bbox[3] * cfg.depth_render_shape[0]], dtype=np.float32)
        princpt_depth = np.array([(princpt[0] - bbox[0]) / bbox[2] * cfg.depth_render_shape[1], (princpt[1] - bbox[1]) / bbox[3] * cfg.depth_render_shape[1]], dtype=np.float32)
       
        data = {'kpt_img': kpt_img, 'kpt_valid': kpt_valid, 'depth': depth, 'smplx_param': smplx_param, 'cam_param': {'focal': focal, 'princpt': princpt}, 'cam_param_kpt': {'focal': focal_kpt, 'princpt': princpt_kpt}, 'cam_param_depth': {'focal': focal_depth, 'princpt': princpt_depth}, 'frame_idx': frame_idx}
        return data
