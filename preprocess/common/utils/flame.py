import numpy as np
import torch
import os.path as osp
from config import cfg
import smplx

class FLAME(object):
    def __init__(self):
        self.shape_param_dim = 100
        self.expr_param_dim = 50
        self.layer_arg = {'create_betas': False, 'create_expression': False, 'create_global_orient': False, 'create_neck_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_transl': False}
        self.layer = smplx.create(cfg.human_model_path, 'flame', gender='neutral', num_betas=self.shape_param_dim, num_expression_coeffs=self.expr_param_dim, use_face_contour=True, **self.layer_arg)
        self.vertex_num = 5023
        self.face = self.layer.faces.astype(np.int64)

        # joint
        self.joint = {
                'num': 5,
                'name': ('Neck', 'Head', 'Jaw', 'L_Eye', 'R_Eye'),
                'root_idx': 0
                }

        # keypoint
        self.kpt = {
                'num': 76,
                'name': ['Neck', 'Head', 'Jaw', 'L_Eye', 'R_Eye'] +  ['Face_' + str(i) for i in range(1,69)] + ['Nose', 'L_Ear', 'R_Ear'],
                'root_idx': 0
                }

        # vertex idxs
        self.nose_vertex_idx = 3526
        self.lear_vertex_idx = 160
        self.rear_vertex_idx = 1167
        
flame = FLAME()

