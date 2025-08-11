import os
import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from config import cfg
from utils.smpl_x import smpl_x
from utils.preprocessing import load_img
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.data_split = data_split
        self.root_path = osp.join('..', 'data', 'subjects', cfg.subject_id)
        self.transform = transform
        self.smplx_params, self.frame_idx_list = self.load_data()
        self.load_id_info()
        self.kpt = {
                    'num': 133,
                    'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
                        *['Face_' + str(i) for i in range(52,69)], # face contour
                        *['Face_' + str(i) for i in range(1,52)], # face
                        'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
                        'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
                    }

    def load_data(self):
        
        # define splits
        if cfg.fit_pose_to_test:
            split_list = ['test']
        else:
            split_list = ['captured', 'generated_0', 'generated_1']
      
        # load data
        smplx_params, frame_idx_list = {}, {}
        for split in split_list:
            # smplx parameters
            smplx_params[split] = {}
            smplx_param_path_list = glob(osp.join(self.root_path, split, 'smplx_optimized', 'smplx_params', '*.json'))
            for smplx_param_path in smplx_param_path_list:
                file_name = smplx_param_path.split('/')[-1]
                frame_idx = int(file_name[:-5])
                with open(smplx_param_path) as f:
                    smplx_params[split][frame_idx] = {k: torch.FloatTensor(v) for k,v in json.load(f).items()}
            
            # frame indices
            frame_idx_list[split] = []
            if 'generated' in split:
                with open(osp.join(self.root_path, split, 'frame_idx_list.txt')) as f:
                    frame_idx_list[split] += sorted([int(x) for x in f.readlines()])
            else:
                frame_idx_list[split] += sorted([int(x) for x in smplx_params[split].keys()])

        if self.data_split == 'train':
            # balanced sampling of 'captured' and 'generated' images
            if not cfg.fit_pose_to_test:
                frame_idx_list['captured'] *= sum([len(frame_idx_list[split]) for split in frame_idx_list.keys() if 'generated' in split])

            frame_idx_list = [{'split': split, 'frame_idx': frame_idx} for split in frame_idx_list.keys() for frame_idx in frame_idx_list[split]]
            if cfg.fit_pose_to_test:
                frame_idx_list *= 10
        else:
            frame_idx_list = [{'split': split, 'frame_idx': frame_idx} for split in frame_idx_list.keys() for frame_idx in frame_idx_list[split]]
        
        return smplx_params, frame_idx_list
    
    def load_id_info(self):
        ## load ID parameters
        with open(osp.join(self.root_path, 'captured', 'smplx_optimized', 'shape_param.json')) as f:
            shape_param = torch.FloatTensor(json.load(f)).cuda()
        with open(osp.join(self.root_path, 'captured', 'smplx_optimized', 'face_offset.json')) as f:
            face_offset = torch.FloatTensor(json.load(f)).cuda()
        with open(osp.join(self.root_path, 'captured', 'smplx_optimized', 'joint_offset.json')) as f:
            joint_offset = torch.FloatTensor(json.load(f)).cuda()
        smpl_x.set_id_info(shape_param, face_offset, joint_offset)

        ## load unwrapped texture (smplx)
        # from 'captured' images
        texture_path = osp.join(self.root_path, 'captured', 'smplx_optimized', 'smplx_texture_inpaint.png')
        texture_cap = torch.FloatTensor(cv2.imread(texture_path)[:,:,::-1].copy().transpose(2,0,1).astype(np.float32)/255)
        seg_path = osp.join(self.root_path, 'captured', 'smplx_optimized', 'smplx_seg_inpaint.png')
        seg_cap = torch.FloatTensor(cv2.imread(seg_path)[:,:,::-1].copy().transpose(2,0,1).astype(np.float32)/255)
        mask_path = osp.join(self.root_path, 'captured', 'smplx_optimized', 'smplx_texture_mask.png')
        mask_cap = torch.FloatTensor((cv2.imread(mask_path) > 0)[:,:,0,None].transpose(2,0,1).astype(np.float32))

        # from 'generated' images
        texture_path = osp.join(self.root_path, 'generated_0', 'smplx_optimized', 'smplx_texture_inpaint.png')
        texture_gen = torch.FloatTensor(cv2.imread(texture_path)[:,:,::-1].copy().transpose(2,0,1).astype(np.float32)/255)
        seg_path = osp.join(self.root_path, 'generated_0', 'smplx_optimized', 'smplx_seg_inpaint.png')
        seg_gen = torch.FloatTensor(cv2.imread(seg_path)[:,:,::-1].copy().transpose(2,0,1).astype(np.float32)/255)
        mask_path = osp.join(self.root_path, 'generated_0', 'smplx_optimized', 'smplx_texture_mask.png')
        mask_gen = torch.FloatTensor((cv2.imread(mask_path) > 0)[:,:,0,None].transpose(2,0,1).astype(np.float32))

        # combine (prioritize texture from 'captured' images)
        texture = texture_cap * mask_cap + texture_gen * (1 - mask_cap)
        smpl_x.set_texture(texture, texture_gen, seg_gen)
        
    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        split, frame_idx = self.frame_idx_list[idx]['split'], self.frame_idx_list[idx]['frame_idx']

        # load image
        img_path = osp.join(self.root_path, split, 'images', str(frame_idx) + '.png')
        img = load_img(img_path)
        img = self.transform(img.astype(np.float32))/255.

        # load mask
        mask_path = osp.join(self.root_path, split, 'masks', str(frame_idx) + '.png')
        mask = cv2.imread(mask_path)[:,:,0,None] / 255.
        mask = self.transform((mask > 0.5).astype(np.float32))

        # load albedo
        if split == 'captured':
            albedo_path = osp.join(self.root_path, split, 'image_intrinsics', str(frame_idx) + '_albedo.png')
            albedo = load_img(albedo_path)
            albedo = self.transform(albedo.astype(np.float32))/255.
        else:
            albedo = torch.ones_like(img)

        # load face image and albedo
        if split == 'captured':
            # image
            face_patch_path = osp.join(self.root_path, split, 'face_patches', 'images', str(frame_idx) + '.png')
            face_patch = load_img(face_patch_path)
            face_height, face_width, _ = face_patch.shape
            assert (face_height,face_width) == cfg.face_patch_shape
            face_patch = self.transform(face_patch.astype(np.float32))/255.
            # bbox
            face_bbox_path = osp.join(self.root_path, split, 'face_patches', 'bbox', str(frame_idx) + '.json')
            with open(face_bbox_path) as f:
                face_bbox = np.array(json.load(f), dtype=np.float32) # xywh
        else:
            face_patch = torch.ones((3, cfg.face_patch_shape[0], cfg.face_patch_shape[1])).float()
            face_bbox = np.array([0,0,1,1], dtype=np.float32)

        # load depth
        depth_path = osp.join(self.root_path, split, 'depths', str(frame_idx) + '.npy')
        depth = np.load(depth_path)[:,:,None]
        depth_mask = np.load(depth_path[:-4] + '_mask.npy')[:,:,None]
        depth = (depth * depth_mask).transpose(2,0,1) # 1, img_height, img_width

        # load normal
        normal_path = osp.join(self.root_path, split, 'normals', str(frame_idx) + '.npy')
        normal = np.load(normal_path) # [-1,1]
        normal = (normal + 1)/2. # [0,1]
        normal_mask = np.load(normal_path[:-4] + '_mask.npy')[:,:,None]
        normal = (normal * normal_mask).transpose(2,0,1) # 3, img_height, img_width

        # load segmentation
        seg_path = osp.join(self.root_path, split, 'segs', str(frame_idx) + '.png')
        seg = load_img(seg_path)
        seg = self.transform(seg.astype(np.float32))/255.
        seg_mask = torch.FloatTensor(np.load(seg_path[:-4] + '.npy'))[None,:,:]
        seg = seg*seg_mask

        # load camera parameter
        cam_param_path = osp.join(self.root_path, split, 'cam_params', str(frame_idx) + '.json')
        with open(cam_param_path) as f:
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in json.load(f).items()}
        cam_param_face = {'R': cam_param['R'], 't': cam_param['t'], \
                        'focal': np.array([cam_param['focal'][0]/face_bbox[2]*cfg.face_patch_shape[1], cam_param['focal'][1]/face_bbox[3]*cfg.face_patch_shape[0]], dtype=np.float32),
                        'princpt': np.array([(cam_param['princpt'][0]-face_bbox[0])/face_bbox[2]*cfg.face_patch_shape[1], (cam_param['princpt'][1]-face_bbox[1])/face_bbox[3]*cfg.face_patch_shape[0]], dtype=np.float32)}

        data = {'img': img, 'mask': mask, 'albedo': albedo, 'face_patch': face_patch, 'depth': depth, 'normal': normal, 'seg': seg, 'cam_param': cam_param, 'cam_param_face': cam_param_face, 'split': split, 'frame_idx': frame_idx}
        return data

