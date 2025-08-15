import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.loss import CoordLoss, PoseLoss, DepthLoss, PoseReg, LaplacianReg, FaceOffsetSymmetricReg, JointOffsetSymmetricReg
from nets.layer import rasterize_xy, render_vertex_texture
from utils.smpl_x import smpl_x
from utils.flame import flame
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import math
from config import cfg

class Model_w_id_opt(nn.Module):
    def __init__(self):
        super(Model_w_id_opt, self).__init__()
        self.smplx_layer = copy.deepcopy(smpl_x.layer).cuda()
        self.flame_layer = copy.deepcopy(flame.layer).cuda()
        self.coord_loss = CoordLoss()
        self.pose_loss = PoseLoss()
        self.depth_loss = DepthLoss()
        self.pose_reg = PoseReg()
        self.lap_reg = LaplacianReg(flame.vertex_num, flame.face)
        self.face_offset_sym_reg = FaceOffsetSymmetricReg()
        self.joint_offset_sym_reg = JointOffsetSymmetricReg()

    def process_input_smplx_param(self, smplx_param):
        out = {}

        # rotation 6d -> axis angle
        for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose']:
            out[key] = matrix_to_axis_angle(rotation_6d_to_matrix(smplx_param[key]))

        # others
        out['trans'] = smplx_param['trans']
        out['expr'] = smplx_param['expr']
        out['shape'] = smplx_param['shape']
        out['face_offset'] = smplx_param['face_offset']
        out['joint_offset'] = smplx_param['joint_offset']
        return out

    def process_input_flame_param(self, flame_param):
        out = {}

        # rotation 6d -> axis angle
        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose']:
            out[key] = matrix_to_axis_angle(rotation_6d_to_matrix(flame_param[key]))

        # others
        out['trans'] = flame_param['trans']
        out['expr'] = flame_param['expr']
        out['shape'] = flame_param['shape']
        return out

    def get_smplx_coord(self, smplx_param, cam_param, use_pose=True, use_expr=True, use_face_offset=True, use_joint_offset=True, root_rel=False):
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

        if use_face_offset:
            face_offset = smpl_x.get_face_offset(smplx_param['face_offset'])
        else:
            face_offset = None
 
        if use_joint_offset:
            joint_offset = smpl_x.get_joint_offset(smplx_param['joint_offset'])
        else:
            joint_offset = None
       
        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose.detach(), leye_pose=leye_pose.detach(), reye_pose=reye_pose.detach(), left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr.detach(), betas=smplx_param['shape'], face_offset=face_offset, joint_offset=joint_offset) # detach jaw_pose, leye_pose, reye_pose, and expr as they are optimized by flame layer

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

    def get_flame_coord(self, flame_param, cam_param, use_pose=True, use_expr=True):
        if use_pose:
            root_pose = flame_param['root_pose']
            neck_pose = flame_param['neck_pose']
            jaw_pose = flame_param['jaw_pose']
            leye_pose = flame_param['leye_pose']
            reye_pose = flame_param['reye_pose']
        else:
            root_pose = torch.zeros_like(flame_param['root_pose'])
            neck_pose = torch.zeros_like(flame_param['neck_pose'])
            jaw_pose = torch.zeros_like(flame_param['jaw_pose'])
            leye_pose = torch.zeros_like(flame_param['leye_pose'])
            reye_pose = torch.zeros_like(flame_param['reye_pose'])
        
        if use_expr:
            expr = flame_param['expr']
        else:
            expr = torch.zeros_like(flame_param['expr'])

        output = self.flame_layer(global_orient=root_pose, neck_pose=neck_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, expression=expr, betas=flame_param['shape'])

        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        kpt_cam = output.joints
        nose = mesh_cam[:,flame.nose_vertex_idx,:]
        lear = mesh_cam[:,flame.lear_vertex_idx,:]
        rear = mesh_cam[:,flame.rear_vertex_idx,:]
        kpt_cam = torch.cat((kpt_cam, nose[:,None,:], lear[:,None,:], rear[:,None,:]),1) # follow flame.kpt['name']
        root_cam = kpt_cam[:,flame.kpt['root_idx'],:]
        mesh_cam = mesh_cam - root_cam[:,None,:] + flame_param['trans'][:,None,:]
        kpt_cam = kpt_cam - root_cam[:,None,:] + flame_param['trans'][:,None,:]
        
        if cam_param is not None:
            # project to the 2D space
            x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
            y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
            kpt_proj = torch.stack((x,y),2) 
            return mesh_cam, kpt_cam, kpt_proj
        else:
            return mesh_cam, kpt_cam
    
    def check_face_visibility(self, face_mesh, leye, reye):
        center = face_mesh.mean(1)
        eye = (leye + reye)/2.

        eye_vec = eye - center
        cam_vec = center - 0

        eye_vec = F.normalize(torch.stack((eye_vec[:,0], eye_vec[:,2]),1), p=2, dim=1)
        cam_vec = F.normalize(torch.stack((cam_vec[:,0], cam_vec[:,2]),1), p=2, dim=1)

        dot_prod = torch.sum(eye_vec * cam_vec, 1)
        face_valid = dot_prod < math.cos(math.pi/4*3)
        return face_valid

    def get_smplx_full_pose(self, smplx_param):
        pose = torch.cat((smplx_param['root_pose'][:,None,:], smplx_param['body_pose'], smplx_param['jaw_pose'][:,None,:], smplx_param['leye_pose'][:,None,:], smplx_param['reye_pose'][:,None,:], smplx_param['lhand_pose'], smplx_param['rhand_pose']),1) # follow smpl_x.joint['name']
        return pose
 
    def forward(self, smplx_inputs, flame_inputs, data, return_output):
        smplx_inputs = self.process_input_smplx_param(smplx_inputs) 
        flame_inputs = self.process_input_flame_param(flame_inputs)
        batch_size = smplx_inputs['root_pose'].shape[0]
       
        # get coordinates from optimizable parameters
        smplx_mesh_cam, smplx_kpt_cam, smplx_root_cam, smplx_kpt_proj = self.get_smplx_coord(smplx_inputs, data['cam_param_kpt'])
        smplx_mesh_cam_wo_fo, smplx_kpt_cam_wo_fo, smplx_root_cam_wo_fo, smplx_kpt_proj_wo_fo = self.get_smplx_coord(smplx_inputs, data['cam_param_kpt'], use_face_offset=False)
        flame_mesh_cam, flame_kpt_cam, flame_kpt_proj = self.get_flame_coord(flame_inputs, data['cam_param_kpt'])
        
        # get coordinates from zero pose
        smplx_mesh_wo_pose_wo_expr, _, _ = self.get_smplx_coord(smplx_inputs, cam_param=None, use_pose=False, use_expr=False, root_rel=True)
        flame_mesh_wo_pose_wo_expr, _ = self.get_flame_coord(flame_inputs, cam_param=None, use_pose=False, use_expr=False)
        
        with torch.no_grad():
            # get coordinates from the initial parameters
            data['smplx_param']['shape'] = smplx_inputs['shape'].clone().detach()
            data['smplx_param']['expr'] = smplx_inputs['expr'].clone().detach()
            data['smplx_param']['trans'] = smplx_inputs['trans'].clone().detach()
            data['smplx_param']['joint_offset'] = smplx_inputs['joint_offset'].clone().detach()
            smplx_mesh_cam_init, smplx_kpt_cam_init, _, _ = self.get_smplx_coord(data['smplx_param'], data['cam_param_kpt'], use_face_offset=False)

            # check face visibility
            face_valid = self.check_face_visibility(smplx_mesh_cam_init[:,smpl_x.face_vertex_idx,:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('L_Eye'),:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('R_Eye'),:])
            face_valid = face_valid * data['flame_valid']
        
        # loss functions
        loss = {}
        weight = torch.ones_like(smplx_kpt_proj)
        weight[:,[i for i in range(smpl_x.kpt['num']) if 'Face' in smpl_x.kpt['name'][i]],:] = 0
        weight[face_valid,:,:] = 1 # do not use 2D loss if face is not visible
        weight[:,smpl_x.kpt['name'].index('R_Eye'),:] *= 10
        weight[:,smpl_x.kpt['name'].index('L_Eye'),:] *= 10
        weight[:,[smpl_x.kpt['name'].index(name) for name in ['Face_' + str(i) for i in range(52,69)]],:] *= 0.1
        loss['smplx_kpt_proj_wo_fo'] = self.coord_loss(smplx_kpt_proj_wo_fo, data['kpt_img'], data['kpt_valid'], smplx_kpt_cam.detach()) * weight
        loss['flame_kpt_proj'] = torch.abs(flame_kpt_proj - data['kpt_img'][:,smpl_x.kpt['part_idx']['face'],:]) * data['kpt_valid'][:,smpl_x.kpt['part_idx']['face'],:] * weight[:,smpl_x.kpt['part_idx']['face'],:]
        if cfg.use_depthmap_loss:
            # rasterize depth map and compute depthmap loss
            fragments = rasterize_xy(smplx_mesh_cam_wo_fo, smpl_x.face, data['cam_param_depth'], cfg.depth_render_shape)
            is_face = torch.zeros((smpl_x.vertex_num)).index_fill(0, torch.LongTensor(smpl_x.face_vertex_idx), 1.0).cuda()[None,:].repeat(batch_size,1)
            face_mask = render_vertex_texture(fragments, is_face, smpl_x.face).detach()
            smplx_depth = fragments.zbuf[:,None,:,:,0]
            flame_depth = rasterize_xy(flame_mesh_cam, flame.face, data['cam_param_depth'], cfg.depth_render_shape).zbuf[:,None,:,:,0]
            loss['smplx_depth'] = self.depth_loss(smplx_depth, data['depth']) * 0.01
            loss['flame_depth'] = self.depth_loss(flame_depth*face_mask, data['depth']*face_mask) * 0.01
        loss['smplx_shape_reg'] = smplx_inputs['shape'] ** 2 * 0.01
        loss['smplx_mesh'] = torch.abs((smplx_mesh_cam_wo_fo - smplx_kpt_cam_wo_fo[:,smpl_x.kpt['root_idx'],None,:]) - \
                                        (smplx_mesh_cam_init - smplx_kpt_cam_init[:,smpl_x.kpt['root_idx'],None,:])) * 0.1 
        smplx_input_pose = self.get_smplx_full_pose(smplx_inputs)
        smplx_init_pose = self.get_smplx_full_pose(data['smplx_param'])
        loss['smplx_pose'] = self.pose_loss(smplx_input_pose, smplx_init_pose) * 0.1
        loss['smplx_pose_reg'] = self.pose_reg(smplx_input_pose)

        loss['flame_pose'] = self.pose_loss(flame_inputs['jaw_pose'], data['flame_param']['jaw_pose']) * 0.01
        loss['flame_shape'] = (flame_inputs['shape'] - data['flame_param']['shape']) ** 2 * 0.01
        loss['flame_expr'] = (flame_inputs['expr'] - data['flame_param']['expr']) ** 2 * 0.01

        is_not_neck = torch.ones((1,flame.vertex_num,1)).float().cuda()
        is_not_neck[:,flame.layer.lbs_weights.argmax(1)==flame.joint['root_idx'],:] = 0
        loss['smplx_to_flame_v2v_wo_pose_expr'] = torch.abs(\
                (smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:] - smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:].mean(1)[:,None,:]) - \
                (flame_mesh_wo_pose_wo_expr - flame_mesh_wo_pose_wo_expr.mean(1)[:,None,:]).detach()) * is_not_neck * 10
        loss['smplx_to_flame_lap'] = self.lap_reg(smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:], flame_mesh_wo_pose_wo_expr.detach()) * is_not_neck * 100000

        is_neck = torch.zeros((1,flame.vertex_num,1)).float().cuda()
        is_neck[:,flame.layer.lbs_weights.argmax(1)==flame.joint['root_idx'],:] = 1
        loss['face_offset_reg'] = smplx_inputs['face_offset'] ** 2 * is_neck * 1000
        weight = torch.ones((1,smpl_x.joint['num'],1)).float().cuda()
        loss['joint_offset_reg'] = smplx_inputs['joint_offset'] ** 2 * 100 * weight
        loss['face_offset_sym_reg'] = self.face_offset_sym_reg(smplx_inputs['face_offset'])
        loss['joint_offset_sym_reg'] = self.joint_offset_sym_reg(smplx_inputs['joint_offset'])
        
        if not return_output:
            return loss, None
        else:
            # for the visualization
            smplx_mesh_cam_wo_jo, _, _ = self.get_smplx_coord(smplx_inputs, cam_param=None, use_joint_offset=False)
            smplx_mesh_wo_pose_wo_expr_wo_fo, _, _ = self.get_smplx_coord(smplx_inputs, cam_param=None, use_pose=False, use_expr=False, use_face_offset=False, root_rel=True)

            # translation alignment
            offset = -flame_mesh_wo_pose_wo_expr.mean(1) + smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:].mean(1)
            flame_mesh_wo_pose_wo_expr = flame_mesh_wo_pose_wo_expr + offset[:,None,:]
            
            # outputs
            out = {}
            out['smplx_mesh_cam'] = smplx_mesh_cam
            out['smplx_mesh_cam_wo_jo'] = smplx_mesh_cam_wo_jo
            out['smplx_mesh_cam_wo_fo'] = smplx_mesh_cam_wo_fo
            out['smplx_trans'] = smplx_inputs['trans'] - smplx_root_cam
            out['flame_mesh_cam'] = flame_mesh_cam
            out['smplx_mesh_wo_pose_wo_expr'] = smplx_mesh_wo_pose_wo_expr
            out['smplx_mesh_wo_pose_wo_expr_wo_fo'] = smplx_mesh_wo_pose_wo_expr_wo_fo
            out['flame_mesh_wo_pose_wo_expr'] = flame_mesh_wo_pose_wo_expr
            return loss, out
 
class Model_wo_id_opt(nn.Module):
    def __init__(self):
        super(Model_wo_id_opt, self).__init__()
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
        out['face_offset'] = smplx_param['face_offset']
        out['joint_offset'] = smplx_param['joint_offset']
        return out

    def get_smplx_coord(self, smplx_param, cam_param, use_pose=True, use_expr=True, use_face_offset=True, use_joint_offset=True, root_rel=False):
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

        if use_face_offset:
            face_offset = smpl_x.get_face_offset(smplx_param['face_offset'])
        else:
            face_offset = None
 
        if use_joint_offset:
            joint_offset = smpl_x.get_joint_offset(smplx_param['joint_offset'])
        else:
            joint_offset = None
       
        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=smplx_param['shape'], face_offset=face_offset, joint_offset=joint_offset) 
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
            data['smplx_param']['joint_offset'] = smplx_inputs['joint_offset'].clone().detach()
            smplx_mesh_cam_init, smplx_kpt_cam_init, _, _ = self.get_smplx_coord(data['smplx_param'], data['cam_param_kpt'], use_face_offset=False)

        # loss functions
        loss = {}
        loss['smplx_kpt_proj'] = self.coord_loss(smplx_kpt_proj, data['kpt_img'], data['kpt_valid'], smplx_kpt_cam.detach())
        if cfg.use_depthmap_loss:
            # rasterize depth map and compute depthmap loss
            depth = rasterize_xy(smplx_mesh_cam, smpl_x.face, data['cam_param_depth'], cfg.depth_render_shape).zbuf[:,None,:,:,0]
            loss['depth'] = self.depth_loss(depth, data['depth']) * 0.01
        loss['smplx_mesh'] = torch.abs((smplx_mesh_cam - smplx_kpt_cam[:,smpl_x.kpt['root_idx'],None,:]) - \
                                        (smplx_mesh_cam_init - smplx_kpt_cam_init[:,smpl_x.kpt['root_idx'],None,:])) * 0.1 
        smplx_input_pose = self.get_smplx_full_pose(smplx_inputs)
        smplx_init_pose = self.get_smplx_full_pose(data['smplx_param'])
        loss['smplx_pose'] = self.pose_loss(smplx_input_pose, smplx_init_pose) * 0.1
        loss['smplx_pose_reg'] = self.pose_reg(smplx_input_pose)
        loss['smplx_expr'] = (smplx_inputs['expr'] - data['flame_param']['expr']) ** 2 * 0.01
        
        if not return_output:
            return loss, None
        else:
            # for the visualization
            smplx_mesh_wo_pose_wo_expr_wo_fo, _, _ = self.get_smplx_coord(smplx_inputs, cam_param=None, use_pose=False, use_expr=False, use_face_offset=False, root_rel=True)

            # outputs
            out = {}
            out['smplx_mesh_cam'] = smplx_mesh_cam
            out['smplx_trans'] = smplx_inputs['trans'] - smplx_root_cam
            return loss, out

def get_model(id_opt):
    if id_opt:
        model = Model_w_id_opt()
    else:
        model = Model_wo_id_opt()
    return model

