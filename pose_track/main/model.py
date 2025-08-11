import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.loss import CoordLoss, PoseLoss, DepthLoss, PoseReg
from nets.layer import rasterize_xy
from utils.smpl_x import smpl_x
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import math
from config import cfg

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.smplx_layer = copy.deepcopy(smpl_x.layer).cuda()
        self.coord_loss = CoordLoss()
        self.pose_loss = PoseLoss()
        self.depth_loss = DepthLoss()
        self.pose_reg = PoseReg()

    def process_input_smplx_param(self, smplx_param):
        out = {}

        # rotation 6d -> axis angle
        for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose']:
            out[key] = matrix_to_axis_angle(rotation_6d_to_matrix(smplx_param[key]))

        # others
        out['trans'] = smplx_param['trans']
        out['expr'] = smplx_param['expr']
        out['shape'] = smplx_param['shape']
        return out

    def get_smplx_coord(self, smplx_param, cam_param, use_pose=True, use_expr=True, root_rel=False):
        batch_size = smplx_param['root_pose'].shape[0]
       
        if use_pose:
            root_pose = smplx_param['root_pose']
            body_pose = smplx_param['body_pose']
            jaw_pose = smplx_param['jaw_pose']
            leye_pose = smplx_param['leye_pose']
            reye_pose = smplx_param['reye_pose']
            lhand_pose = smplx_param['lhand_pose']
            rhand_pose = smplx_param['rhand_pose']
        else:
            root_pose = torch.zeros_like(smplx_param['root_pose'])
            body_pose = torch.zeros_like(smplx_param['body_pose'])
            jaw_pose = torch.zeros_like(smplx_param['jaw_pose'])
            leye_pose = torch.zeros_like(smplx_param['leye_pose'])
            reye_pose = torch.zeros_like(smplx_param['reye_pose'])
            lhand_pose = torch.zeros_like(smplx_param['lhand_pose'])
            rhand_pose = torch.zeros_like(smplx_param['rhand_pose'])
        
        if use_expr:
            expr = smplx_param['expr']
        else:
            expr = torch.zeros_like(smplx_param['expr'])

        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=smplx_param['shape']) 

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        kpt_cam = output.joints[:,smpl_x.kpt['idx'],:]
        root_cam = kpt_cam[:,smpl_x.kpt['root_idx'],:]
        mesh_cam = mesh_cam - root_cam[:,None,:] + smplx_param['trans'][:,None,:]
        kpt_cam = kpt_cam - root_cam[:,None,:] + smplx_param['trans'][:,None,:]
        
        # project to the 2D space
        if cam_param is not None:
            x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
            y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
            kpt_proj = torch.stack((x,y),2)

        if root_rel:
            mesh_cam = mesh_cam - smplx_param['trans'][:,None,:]
            kpt_cam = kpt_cam - smplx_param['trans'][:,None,:]

        if cam_param is not None:
            return mesh_cam, kpt_cam, root_cam, kpt_proj
        else:
             return mesh_cam, kpt_cam, root_cam

    def get_smplx_full_pose(self, smplx_param):
        pose = torch.cat((smplx_param['root_pose'][:,None,:], smplx_param['body_pose'], smplx_param['jaw_pose'][:,None,:], smplx_param['leye_pose'][:,None,:], smplx_param['reye_pose'][:,None,:], smplx_param['lhand_pose'], smplx_param['rhand_pose']),1) # follow smpl_x.joint['name']
        return pose
 
    def forward(self, smplx_inputs, data, return_output):
        smplx_inputs = self.process_input_smplx_param(smplx_inputs) 
       
        # get coordinates from optimizable parameters
        smplx_mesh_cam, smplx_kpt_cam, smplx_root_cam, smplx_kpt_proj = self.get_smplx_coord(smplx_inputs, data['cam_param_kpt'])
        
        with torch.no_grad():
            # get coordinates from the initial parameters
            data['smplx_param']['shape'] = smplx_inputs['shape'].clone().detach()
            data['smplx_param']['expr'] = smplx_inputs['expr'].clone().detach()
            data['smplx_param']['trans'] = smplx_inputs['trans'].clone().detach()
            smplx_mesh_cam_init, smplx_kpt_cam_init, _, _ = self.get_smplx_coord(data['smplx_param'], data['cam_param_kpt'])

        # loss functions
        loss = {}
        loss['smplx_kpt_proj'] = self.coord_loss(smplx_kpt_proj, data['kpt_img'], data['kpt_valid'], smplx_kpt_cam.detach())
        if cfg.use_depthmap_loss:
            # rasterize depth map and compute depthmap loss
            smplx_depth = rasterize_xy(smplx_mesh_cam, smpl_x.face, data['cam_param_depth'], cfg.depth_render_shape).zbuf[:,None,:,:,0]
            loss['depth'] = self.depth_loss(smplx_depth, data['depth']) * 0.01
        loss['smplx_shape_reg'] = smplx_inputs['shape'] ** 2 * 0.01
        loss['smplx_mesh'] = torch.abs((smplx_mesh_cam - smplx_kpt_cam[:,smpl_x.kpt['root_idx'],None,:]) - \
                                        (smplx_mesh_cam_init - smplx_kpt_cam_init[:,smpl_x.kpt['root_idx'],None,:])) * 0.1 
        smplx_input_pose = self.get_smplx_full_pose(smplx_inputs)
        smplx_init_pose = self.get_smplx_full_pose(data['smplx_param'])
        loss['smplx_pose'] = self.pose_loss(smplx_input_pose, smplx_init_pose) * 0.1
        loss['smplx_pose_reg'] = self.pose_reg(smplx_input_pose)

        if not return_output:
            return loss, None
        else:
            # outputs
            out = {}
            out['smplx_mesh_cam'] = smplx_mesh_cam
            out['smplx_trans'] = smplx_inputs['trans'] - smplx_root_cam
            return loss, out
 
def get_model():
    model = Model()
    return model
