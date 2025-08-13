import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from config import cfg
from utils.smpl_x import smpl_x
from utils.flame import flame
from utils.preprocessing import load_img, get_bbox, set_aspect_ratio, get_patch_img
from utils.transforms import change_kpt_name
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from pytorch3d.ops import corresponding_points_alignment
import json
import math

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.root_path = osp.join('..', 'data', 'subjects', cfg.subject_id, cfg.split)
        self.transform = transform
        self.kpt = {
                    'num': 133,
                    'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
                        *['Face_' + str(i) for i in range(52,69)], # face contour
                        *['Face_' + str(i) for i in range(1,52)], # face
                        'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
                        'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
                    }
        self.cam_params, self.kpts, self.depth_paths, self.smplx_params, self.flame_params, self.flame_shape_param, self.frame_idx_list = self.load_data()
        self.get_flame_root_init() # get initial flame root pose and translation

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
            # load keypoints from super-resolution face image
            if cfg.split == 'captured':
                face_kpt_path = osp.join(self.root_path, 'face_patches', 'keypoints_whole_body', str(frame_idx) + '.json')
                with open(face_kpt_path) as f:
                    face_kpt = np.array(json.load(f), dtype=np.float32)
                face_bbox_path = osp.join(self.root_path, 'face_patches', 'bbox', str(frame_idx) + '.json')
                with open(face_bbox_path) as f:
                    face_bbox = np.array(json.load(f), dtype=np.float32)
                face_kpt[:,0] = face_kpt[:,0] / cfg.face_patch_shape[1] * face_bbox[2] + face_bbox[0]
                face_kpt[:,1] = face_kpt[:,1] / cfg.face_patch_shape[0] * face_bbox[3] + face_bbox[1]
                face_kpt_names = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear'] + [name for name in self.kpt['name'] if 'Face_' in name]
                for name in face_kpt_names:
                    kpt[self.kpt['name'].index(name),:] = face_kpt[self.kpt['name'].index(name),:]
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

        # load initial flame parameters 
        flame_params, flame_shape_param = {}, None
        flame_param_path_list = glob(osp.join(self.root_path, 'flame_init', 'flame_params', '*.json'))
        for flame_param_path in flame_param_path_list:
            frame_idx = int(flame_param_path.split('/')[-1][:-5])
            with open(flame_param_path) as f:
                flame_param = json.load(f)
            if not flame_param['is_valid']:
                for k in flame_param.keys():
                    if 'pose' in k:
                        flame_param[k] = torch.zeros((3)).float() # dummy
                flame_param['expr'] = torch.zeros((flame.expr_param_dim)).float() # dummy
            for k,v in flame_param.items():
                if k == 'is_valid':
                    continue
                else:
                    flame_param[k] = torch.FloatTensor(v)
            flame_params[frame_idx] = flame_param
        with open(osp.join(self.root_path, 'flame_init', 'shape_param.json')) as f:
            flame_shape_param = torch.FloatTensor(json.load(f))

        frame_idx_list = [int(x) for x in cam_params.keys()]
        return cam_params, kpts, depth_paths, smplx_params, flame_params, flame_shape_param, frame_idx_list

    def get_flame_root_init(self):
        for i in range(len(self.frame_idx_list)):
            frame_idx = self.frame_idx_list[i]
            
            smplx_root_pose = axis_angle_to_matrix(self.smplx_params[frame_idx]['root_pose'])
            smplx_root_trans = self.smplx_params[frame_idx]['trans']
            smplx_init = torch.matmul(smplx_root_pose, smpl_x.layer.v_template.permute(1,0)).permute(1,0)
            smplx_init = (smplx_init - smplx_init.mean(0)[None,:] + smplx_root_trans[None,:])[smpl_x.face_vertex_idx,:]
            
            # get initial root pose and translation with the rigid alignment
            flame_init = flame.layer.v_template
            RTs = corresponding_points_alignment(flame_init[None], smplx_init[None])
            R = RTs.R.permute(0,2,1)[0]
            flame_init = torch.matmul(R, flame_init.permute(1,0)).permute(1,0)
            self.flame_params[frame_idx]['root_pose'] = matrix_to_axis_angle(R)
            self.flame_params[frame_idx]['trans'] = -flame_init.mean(0) + smplx_init.mean(0)

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

        # flame parameter
        if np.sum(kpt_valid[smpl_x.kpt['part_idx']['face'],0]) == 0:
            self.flame_params[frame_idx]['is_valid'] = False
        flame_param = self.flame_params[frame_idx]
        flame_param['shape'] = self.flame_shape_param
        flame_valid = flame_param['is_valid']

        # modify intrincis to directly project 3D coordinates to cfg.kpt_proj_shape space
        focal = self.cam_params[frame_idx]['focal']
        princpt = self.cam_params[frame_idx]['princpt']
        focal_kpt = np.array([focal[0] / bbox[2] * cfg.kpt_proj_shape[1], focal[1] / bbox[3] * cfg.kpt_proj_shape[0]], dtype=np.float32)
        princpt_kpt = np.array([(princpt[0] - bbox[0]) / bbox[2] * cfg.kpt_proj_shape[1], (princpt[1] - bbox[1]) / bbox[3] * cfg.kpt_proj_shape[1]], dtype=np.float32)
        focal_depth = np.array([focal[0] / bbox[2] * cfg.depth_render_shape[1], focal[1] / bbox[3] * cfg.depth_render_shape[0]], dtype=np.float32)
        princpt_depth = np.array([(princpt[0] - bbox[0]) / bbox[2] * cfg.depth_render_shape[1], (princpt[1] - bbox[1]) / bbox[3] * cfg.depth_render_shape[1]], dtype=np.float32)
       
        data = {'kpt_img': kpt_img, 'kpt_valid': kpt_valid, 'depth': depth, 'smplx_param': smplx_param, 'flame_param': flame_param, 'flame_valid': flame_valid, 'cam_param': {'focal': focal, 'princpt': princpt}, 'cam_param_kpt': {'focal': focal_kpt, 'princpt': princpt_kpt}, 'cam_param_depth': {'focal': focal_depth, 'princpt': princpt_depth}, 'frame_idx': frame_idx}
        return data
