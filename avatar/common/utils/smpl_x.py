import sys
import numpy as np
import torch
from torch.nn import functional as F
import os.path as osp
from config import cfg
from utils.smplx import smplx
import pickle
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.io import load_obj
from utils.transforms import get_neighbor
from collections import deque
import math

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
        self.face_with_mouth_in = self.add_mouth_in_face()
        self.vertex_uv, self.face_uv = self.load_uv_info()
        with open(osp.join(cfg.human_model_path, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.rhand_vertex_idx = hand_vertex_idx['right_hand']
        self.lhand_vertex_idx = hand_vertex_idx['left_hand']
        self.rhand_face = smplx.create(cfg.human_model_path, 'mano', is_rhand=True).faces.astype(np.int64)
        self.lhand_face = smplx.create(cfg.human_model_path, 'mano', is_rhand=False).faces.astype(np.int64)
        self.expr_vertex_idx = self.get_expr_vertex_idx()

        # joint
        self.joint = {
        'num': 55, # 22 (body joints) + 3 (face joints) + 30 (hand joints)
        'name':
        ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
        'Jaw', 'L_Eye', 'R_Eye', # face joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints
        )
        }
        self.joint['root_idx'] = self.joint['name'].index('Pelvis')
        self.joint['part_idx'] = \
        {'body': range(self.joint['name'].index('Pelvis'), self.joint['name'].index('R_Wrist')+1),
        'face': range(self.joint['name'].index('Jaw'), self.joint['name'].index('R_Eye')+1),
        'lhand': range(self.joint['name'].index('L_Index_1'), self.joint['name'].index('L_Thumb_3')+1),
        'rhand': range(self.joint['name'].index('R_Index_1'), self.joint['name'].index('R_Thumb_3')+1)}
        self.neutral_body_pose = torch.zeros((len(self.joint['part_idx']['body'])-1,3)) # 大 pose in axis-angle representation (body pose without root joint)
        self.neutral_body_pose[0] = torch.FloatTensor([0, 0, 1/3])
        self.neutral_body_pose[1] = torch.FloatTensor([0, 0, -1/3])
        self.neutral_jaw_pose = torch.FloatTensor([1/3, 0, 0])
        self.adj_joint_mask = self.get_adj_joint_mask()
        self.is_expr_boundary = self.get_expr_boundary()
        self.is_mouth_in, self.is_mouth_out = self.get_mouth()

        # keypoint
        self.kpt = {
                'num': 135, # 25 (body joints) + 40 (hand joints) + 70 (face keypoints)
                'name': ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body joints
                         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
                         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
                         'Head', 'Jaw', *['Face_' + str(i) for i in range(1,69)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
                        ),
                'idx': (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints
                    37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand joints
                    52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints
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
        self.kpt['part_idx'] = {
                'face': [self.kpt['name'].index('L_Eye'), self.kpt['name'].index('R_Eye')] + list(range(self.kpt['name'].index('Face_1'), self.kpt['name'].index('Face_51')+1))}


        # subdivider
        self.subdivider_list = self.get_subdivider(2)
        self.face_upsampled = self.subdivider_list[-1]._subdivided_faces.cpu().numpy()
        self.vertex_num_upsampled = int(np.max(self.face_upsampled)+1)

    def get_expr_from_flame(self, smplx_layer):
        flame_layer = smplx.create(cfg.human_model_path, 'flame', gender='neutral', num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim)
        smplx_layer.expr_dirs[self.face_vertex_idx,:,:] = flame_layer.expr_dirs
        return smplx_layer

    def add_mouth_in_face(self):
        lip_vertex_idx = [2844, 2855, 8977, 1740, 1730, 1789, 8953, 2892]
        mouth_in_face = [[0,1,7], [1,2,7], [2, 3,5], [3,4,5], [2,5,6], [2,6,7]]
        face_new = list(self.face)
        for face in mouth_in_face:
            v1, v2, v3 = face
            face_new.append([lip_vertex_idx[v1], lip_vertex_idx[v2], lip_vertex_idx[v3]])
        face_new = np.array(face_new, dtype=np.int64)
        return face_new

    def set_id_info(self, shape_param, face_offset, joint_offset):
        if shape_param is None:
            self.shape_param = torch.zeros((self.shape_param_dim)).float().cuda()
        else:
            self.shape_param = shape_param
        if face_offset is None:
            self.face_offset = torch.zeros((self.vertex_num,3)).float().cuda()
        else:
            self.face_offset = face_offset
        if joint_offset is None:
            self.joint_offset = torch.zeros((self.joint['num'],3)).float().cuda()
        else:
            self.joint_offset = joint_offset

    def get_joint_offset(self, joint_offset):
        weight = torch.ones((1,self.joint['num'],1)).float().cuda()
        weight[:,self.joint['root_idx'],:] = 0
        joint_offset = joint_offset * weight
        return joint_offset

    def get_subdivider(self, subdivide_num):
        vert = self.layer.v_template.float().cuda()
        face = torch.LongTensor(self.face).cuda()
        mesh = Meshes(vert[None,:,:], face[None,:,:])

        subdivider_list = [SubdivideMeshes(mesh)]
        for i in range(subdivide_num-1):
            mesh = subdivider_list[-1](mesh)
            subdivider_list.append(SubdivideMeshes(mesh))
        return subdivider_list

    def upsample_mesh(self, vert, feat_list=None):
        face = torch.LongTensor(self.face).cuda()
        subdivider_list = self.subdivider_list
        mesh = Meshes(vert[None,:,:], face[None,:,:])

        if feat_list is None:
            for subdivider in subdivider_list:
                mesh = subdivider(mesh)
            vert = mesh.verts_list()[0]
            return vert
        else:
            feat_dims = [x.shape[1] for x in feat_list]
            feats = torch.cat(feat_list,1)
            for subdivider in subdivider_list:
                mesh, feats = subdivider(mesh, feats)
            vert = mesh.verts_list()[0]
            feats = feats[0]
            feat_list = torch.split(feats, feat_dims, dim=1)
            return vert, *feat_list

    def lr_idx_to_hr_idx(self, idx):
        # follow 'subdivide_homogeneous' function of https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/subdivide_meshes.html#SubdivideMeshes
        # the low-res part takes first N_lr vertices out of N_hr vertices
        return idx

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
    
    def get_adj_joint_mask(self, ring_size=4):
        # construct a human kinematic graph
        parents = self.layer.parents.tolist()
        graph = {i: [] for i in range(len(parents))}
        for child, parent in enumerate(parents):
            if parent != -1:
                graph[child].append(parent)
                graph[parent].append(child)
        
        # construct a distnace matrix, which represent a distance for all pairs of joints
        N = len(parents)
        distance_matrix = np.full((N, N), np.inf)
        for start in range(N):
            queue = deque([(start, 0)])
            visited = set([start])

            while queue:
                current, dist = queue.popleft()
                distance_matrix[start, current] = dist
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        # get binary mask by taking pairs of joints whose distance is shorter than or equal to ring_size
        adj_joint_mask = torch.FloatTensor(distance_matrix <= ring_size) # smpl_x.joint['num'], smpl_x.joint['num']
        return adj_joint_mask

    def get_mouth(self):
        xyz = self.layer.v_template
        normal = Meshes(verts=self.layer.v_template[None,:,:], faces=torch.LongTensor(self.face)[None,:,:]).verts_normals_packed().reshape(self.vertex_num,3).detach()

        is_mouth_in = torch.zeros((self.vertex_num)).float()
        is_mouth_in[self.expr_vertex_idx] = 1
        is_mouth_in = is_mouth_in * (normal[:,2] < 0) * (xyz[:,2] > (torch.max(xyz[self.expr_vertex_idx,2]) - 0.1)) * (xyz[:,1] < (torch.mean(xyz[self.expr_vertex_idx,1]) - 0.015))
        is_mouth_in = is_mouth_in > 0

        neighbor_idxs, _ = get_neighbor(self.vertex_num, self.face)
        is_mouth_in[neighbor_idxs[torch.nonzero(is_mouth_in)]] = 1
        is_mouth_in[neighbor_idxs[torch.nonzero(is_mouth_in)]] = 1
        is_mouth_out = is_mouth_in.clone()
        is_mouth_out[neighbor_idxs[torch.nonzero(is_mouth_out)]] = 1
        is_mouth_out = (is_mouth_out > 0) * (is_mouth_in == 0)

        return is_mouth_in, is_mouth_out

    def get_expr_boundary(self):
        is_face = 0
        for name in ('Head', 'L_Eye', 'R_Eye', 'Jaw'):
            is_face += torch.max(self.layer.lbs_weights, 1)[1]==self.joint['name'].index(name)
        is_face = is_face > 0

        is_face_expr = torch.zeros((self.vertex_num)).float()
        is_face_expr[self.expr_vertex_idx] = 1
        is_face_wo_expr = is_face.clone()
        is_face_wo_expr[self.expr_vertex_idx] = 0
        face_wo_expr_vertex_idx = torch.nonzero(is_face_wo_expr)

        neighbor_idxs, _ = get_neighbor(self.vertex_num, self.face)
        boundary_vertex_idx = neighbor_idxs[face_wo_expr_vertex_idx]
        is_boundary = torch.zeros((self.vertex_num)).float()
        is_boundary[boundary_vertex_idx] = 1
        for i in range(3):
            is_boundary[neighbor_idxs[torch.nonzero(is_boundary)]] = 1
        is_boundary = (is_face_expr > 0) * (is_boundary > 0)

        return is_boundary

    def load_uv_info(self):
        verts, faces, aux = load_obj(osp.join(cfg.human_model_path, 'smplx', 'smplx_uv', 'smplx_uv.obj'))
        vertex_uv = aux.verts_uvs.numpy().astype(np.float32) # (V`, 2)
        face_uv = faces.textures_idx.numpy().astype(np.int64) # (F, 3). 0-based
        vertex_uv[:,1] = 1 - vertex_uv[:,1]
        return vertex_uv, face_uv

    def set_texture(self, texture, texture_gen, seg):
        if texture is None:
            self.texture = torch.zeros((3,cfg.smplx_uvmap_shape[0],cfg.smplx_uvmap_shape[1])).float().cuda()
        else:
            self.texture = texture.cuda()

        if texture_gen is None:
            self.texture_gen = torch.zeros((3,cfg.smplx_uvmap_shape[0],cfg.smplx_uvmap_shape[1])).float().cuda()
        else:
            self.texture_gen = texture_gen.cuda()

        if seg is None:
            self.seg = torch.zeros((3,cfg.smplx_uvmap_shape[0],cfg.smplx_uvmap_shape[1])).float().cuda()
        else:
            self.seg = seg.cuda()

smpl_x = SMPLX()
