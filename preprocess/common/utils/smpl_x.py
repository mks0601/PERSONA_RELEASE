import numpy as np
import torch
import os.path as osp
from config import cfg
from utils.smplx import smplx
from pytorch3d.io import load_obj
from nets.layer import rasterize_uv
import pickle

class SMPLX(object):
    def __init__(self):
        self.shape_param_dim = 100
        self.expr_param_dim = 50
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        self.layer = smplx.create(cfg.human_model_path, 'smplx', gender='male', num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim, use_pca=False, use_face_contour=True, **self.layer_arg)
        self.face_vertex_idx = np.load(osp.join(cfg.human_model_path, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        self.layer = self.get_expr_from_flame(self.layer) 
        self.vertex_num = 10475
        self.face = self.layer.faces.astype(np.int64)
        self.flip_corr = np.load(osp.join(cfg.human_model_path, 'smplx', 'smplx_flip_correspondences.npz'))
        self.vertex_uv, self.face_uv = self.load_uv_info()

        # joint
        self.joint = {
                'num': 55, # 22 (body joints) + 3 (face joints) + 30 (hand joints)
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
                        'Jaw', 'L_Eye', 'R_Eye', # face joints
                        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
                        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints
                        )
                        }
        self.joint['root_idx'] = self.joint['name'].index('Pelvis')
        self.joint['part_idx'] = {'body': range(self.joint['name'].index('Pelvis'), self.joint['name'].index('R_Wrist')+1),
                                'face': range(self.joint['name'].index('Jaw'), self.joint['name'].index('R_Eye')+1),
                                'lhand': range(self.joint['name'].index('L_Index_1'), self.joint['name'].index('L_Thumb_3')+1),
                                'rhand': range(self.joint['name'].index('R_Index_1'), self.joint['name'].index('R_Thumb_3')+1)
                                }

        # keypoint
        self.kpt = {
                'num': 135, # 25 (body kpts) + 40 (hand kpts) + 70 (face keypoints)
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body kpts
                         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand kpts
                         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand kpts
                         'Head', 'Jaw', *['Face_' + str(i) for i in range(1,69)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
                        ),
                'idx': (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body kpts
                    37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand kpts
                    52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand kpts
                    15,22, # head, jaw
                    76,77,78,79,80,81,82,83,84,85, # eyebrow
                    86,87,88,89, # nose
                    90,91,92,93,94, # below nose
                    95,96,97,98,99,100,101,102,103,104,105,106, # eyes
                    107, # right mouth
                    108,109,110,111,112, # upper mouth
                    113, # left mouth
                    114,115,116,117,118, # lower mouth
                    119, # right lip
                    120,121,122, # upper lip
                    123, # left lip
                    124,125,126, # lower lip
                    127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour
                    )
                }
        self.kpt['root_idx'] = self.kpt['name'].index('Pelvis')
        self.kpt['part_idx'] = {
                'body': range(self.kpt['name'].index('Pelvis'), self.kpt['name'].index('Nose')+1),
                'lhand': range(self.kpt['name'].index('L_Thumb_1'), self.kpt['name'].index('L_Pinky_4')+1),
                'rhand': range(self.kpt['name'].index('R_Thumb_1'), self.kpt['name'].index('R_Pinky_4')+1),
                'face': [self.kpt['name'].index('Neck'), self.kpt['name'].index('Head'), self.kpt['name'].index('Jaw'), self.kpt['name'].index('L_Eye'), self.kpt['name'].index('R_Eye')] + list(range(self.kpt['name'].index('Face_1'), self.kpt['name'].index('Face_68')+1)) + [self.kpt['name'].index('Nose'), self.kpt['name'].index('L_Ear'), self.kpt['name'].index('R_Ear')]}
        
        # face mask in UV space
        self.expr_vertex_idx = self.get_expr_vertex_idx()
        self.face_uv_mask = self.get_face_uv_mask()

    def get_expr_from_flame(self, smplx_layer):
        flame_layer = smplx.create(cfg.human_model_path, 'flame', gender='neutral', num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim)
        smplx_layer.expr_dirs[self.face_vertex_idx,:,:] = flame_layer.expr_dirs
        return smplx_layer
    
    def get_face_offset(self, face_offset):
        batch_size = face_offset.shape[0]
        face_offset_pad = torch.zeros((batch_size,self.vertex_num,3)).float().cuda()
        face_offset_pad[:,self.face_vertex_idx,:] = face_offset
        return face_offset_pad
    
    def get_joint_offset(self, joint_offset):
        weight = torch.ones((1,self.joint['num'],1)).float().cuda()
        weight[:,self.joint['root_idx'],:] = 0
        weight[:,self.joint['name'].index('R_Hip'),:] = 0
        weight[:,self.joint['name'].index('L_Hip'),:] = 0
        joint_offset = joint_offset * weight
        return joint_offset
    
    def load_uv_info(self):
        verts, faces, aux = load_obj(osp.join(cfg.human_model_path, 'smplx', 'smplx_uv', 'smplx_uv.obj'))
        vertex_uv = aux.verts_uvs.numpy().astype(np.float32) # (V`, 2)
        face_uv = faces.textures_idx.numpy().astype(np.int64) # (F, 3). 0-based
        vertex_uv[:,1] = 1 - vertex_uv[:,1]
        return vertex_uv, face_uv

    def get_expr_vertex_idx(self):
        # FLAME 2020 has all vertices of expr_vertex_idx. use FLAME 2019
        with open(osp.join(cfg.human_model_path, 'flame', '2019', 'generic_model.pkl'), 'rb') as f:
            flame_2019 = pickle.load(f, encoding='latin1')
        vertex_idxs = np.where((flame_2019['shapedirs'][:,:,300:300+self.expr_param_dim] != 0).sum((1,2)) > 0)[0] # FLAME.SHAPE_SPACE_DIM == 300

        # exclude neck and eyeball regions
        flame_joints_name = ('Neck', 'Head', 'Jaw', 'L_Eye', 'R_Eye')
        expr_vertex_idx = []
        flame_vertex_num = flame_2019['v_template'].shape[0]
        is_neck_eye = torch.zeros((flame_vertex_num)).float()
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('Neck')] = 1
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('L_Eye')] = 1
        is_neck_eye[flame_2019['weights'].argmax(1)==flame_joints_name.index('R_Eye')] = 1
        for idx in vertex_idxs:
            if is_neck_eye[idx]:
                continue
            expr_vertex_idx.append(idx)

        expr_vertex_idx = np.array(expr_vertex_idx)
        expr_vertex_idx = self.face_vertex_idx[expr_vertex_idx]
        return expr_vertex_idx

    def get_face_uv_mask(self):
        # rasterize self.face to UV space
        vertex_uv = torch.from_numpy(self.vertex_uv).float().cuda()[None,:,:]
        face_uv = torch.from_numpy(self.face_uv).long().cuda()[None,:,:]
        outputs = rasterize_uv(vertex_uv, face_uv, cfg.smplx_uvmap_shape)
        pix_to_face = outputs.pix_to_face # batch_size, cfg.smplx_uvmap_shape[0], cfg.smplx_uvmap_shape[1], faces_per_pixel. invalid: -1

        ## make face UV mask
        face_uv_mask = pix_to_face[0,None,:,:,0].clone() # 1, cfg.smplx_uvmap_shape[0], cfg.smplx_uvmap_shape[1]

        # make eyeball mask (eyeball will be 1)
        is_eye = torch.zeros((self.vertex_num)).float()
        is_eye[self.layer.lbs_weights.argmax(1)==self.joint['name'].index('R_Eye')] = 1
        is_eye[self.layer.lbs_weights.argmax(1)==self.joint['name'].index('L_Eye')] = 1
        for i in range(len(self.face)):
            v0, v1, v2 = self.face[i]
            if is_eye[v0] and is_eye[v1] and is_eye[v2]:
                face_uv_mask[face_uv_mask==i] = -2

        # make face mask (face area with expression will be 1)
        expr_vertex_mask = torch.zeros((self.vertex_num)).float()
        expr_vertex_mask[torch.from_numpy(self.expr_vertex_idx).long()] = 1
        for i in range(len(self.face)):
            v0, v1, v2 = self.face[i]
            if expr_vertex_mask[v0] and expr_vertex_mask[v1] and expr_vertex_mask[v2]:
                face_uv_mask[face_uv_mask==i] = -2

        # make neck mask (neck will be 0)
        is_neck = torch.zeros((self.vertex_num)).float()
        is_neck[self.layer.lbs_weights.argmax(1)==self.joint['name'].index('Neck')] = 1
        for i in range(len(self.face)):
            v0, v1, v2 = self.face[i]
            if is_neck[v0] or is_neck[v1] or is_neck[v2]:
                face_uv_mask[face_uv_mask==i] = -1

        face_uv_mask = (face_uv_mask == -2).float()
        return face_uv_mask

smpl_x = SMPLX()
