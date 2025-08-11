import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from utils.transforms import get_fov, get_view_matrix, get_proj_matrix
from utils.smpl_x import smpl_x
from smplx.lbs import batch_rigid_transform
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from nets.layer import make_linear_layers
from config import cfg
import copy
import os.path as osp

class PERSONA(nn.Module):
    def __init__(self):
        super(PERSONA, self).__init__()
        self.smplx_layer = copy.deepcopy(smpl_x.layer).cuda()
        self.triplane = nn.Parameter(torch.zeros((3,*cfg.triplane_shape)).float().cuda())
        self.mean_offset_offset_net = make_linear_layers([cfg.triplane_shape[0]*3+(len(smpl_x.joint['part_idx']['body'])-1)*6, 128, 128, 128, 3], relu_final=False, use_gn=True)

    def init(self):
        ## static assets
        # upsample mesh and other assets
        xyz, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False)
        skinning_weight = self.smplx_layer.lbs_weights.float()
        expr_dirs = self.smplx_layer.expr_dirs.view(smpl_x.vertex_num,3*smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face_expr = torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda()
        is_rhand[smpl_x.rhand_vertex_idx], is_lhand[smpl_x.lhand_vertex_idx], is_face_expr[smpl_x.expr_vertex_idx] = 1.0, 1.0, 1.0
        is_expr_boundary, is_mouth_in, is_mouth_out = smpl_x.is_expr_boundary[:,None].float().cuda(), smpl_x.is_mouth_in[:,None].float().cuda(), smpl_x.is_mouth_out[:,None].float().cuda()
        _, skinning_weight, expr_dirs, is_rhand, is_lhand, is_face_expr, is_expr_boundary, is_mouth_in, is_mouth_out = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num,3)).float().cuda(), [skinning_weight, expr_dirs, is_rhand, is_lhand, is_face_expr, is_expr_boundary, is_mouth_in, is_mouth_out]) # upsample with dummy vertex

        expr_dirs = expr_dirs.view(smpl_x.vertex_num_upsampled,3,smpl_x.expr_param_dim)
        is_rhand, is_lhand, is_face_expr, is_expr_boundary, is_mouth_in, is_mouth_out = is_rhand[:,0] == 1, is_lhand[:,0] == 1, is_face_expr[:,0] == 1, is_expr_boundary[:,0] == 1, is_mouth_in[:,0] == 1, is_mouth_out[:,0] == 1
        is_eye = 0
        for name in ('R_Eye', 'L_Eye'):
            is_eye += torch.max(skinning_weight, 1)[1]==smpl_x.joint['name'].index(name)
        is_eye = is_eye > 0
        is_face = 0
        for name in ('Head', 'L_Eye', 'R_Eye', 'Jaw'):
            is_face += torch.max(skinning_weight, 1)[1]==smpl_x.joint['name'].index(name)
        is_face = is_face > 0

        self.pos_enc_vert = xyz
        self.skinning_weight_orig = skinning_weight
        self.expr_dirs = expr_dirs
        self.is_rhand = is_rhand
        self.is_lhand = is_lhand
        self.is_face_expr = is_face_expr
        self.is_expr_boundary = is_expr_boundary
        self.is_mouth_in = is_mouth_in
        self.is_mouth_out = is_mouth_out
        self.is_hand = (is_rhand+is_lhand) > 0
        self.is_face = is_face
        self.is_eye = is_eye
        self.register_buffer('shape_param', smpl_x.shape_param)
        self.register_buffer('face_offset', smpl_x.face_offset)
        self.register_buffer('joint_offset', smpl_x.joint_offset)
        
        # load diffused skinning weight 
        field = np.load(osp.join(cfg.sw_path, 'diffused_skinning_weights.npy')) # (D_x, D_y, D_z, smpl_x.joint['num'])
        grid_coords_path_list = [osp.join(cfg.sw_path, 'skinning_grid_coords_' + axis + '.npy') for axis in ['x', 'y', 'z']]
        coord_x, coord_y, coord_z = np.load(grid_coords_path_list[0]), np.load(grid_coords_path_list[1]), np.load(grid_coords_path_list[2]) # each: (D_x,), (D_y,), (D_z,)
        self.skinning_weight = torch.FloatTensor(field).cuda()
        self.skinning_weight_grid = torch.stack([torch.FloatTensor(coord_x), torch.FloatTensor(coord_y), torch.FloatTensor(coord_z)],1).cuda()

        ## optimizable assets
        # initialize scale
        xyz, _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=True)
        points = knn_points(xyz[None,:,:], xyz[None,:,:], K=4, return_nn=True)
        dist = torch.sum((xyz[:,None,:] - points.knn[0,:,1:,:])**2,2).mean(1) # average of distances to top-3 closest points (exactly same as https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/gaussian_model.py#L134)
        dist = torch.clamp_min(dist, 0.0000001)
        self.mean_offset = nn.Parameter(torch.zeros((smpl_x.vertex_num_upsampled,3)).float().cuda())
        self.scale = nn.Parameter(torch.log(torch.sqrt(dist))[:,None]) 
        
        # initialize from unwrapped texture
        uv = torch.zeros((smpl_x.vertex_num,2)).float().cuda()
        uv[smpl_x.face,:] = torch.FloatTensor(smpl_x.vertex_uv[smpl_x.face_uv,:]).cuda()
        rgb = F.grid_sample(smpl_x.texture[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        rgb_aux = F.grid_sample(smpl_x.texture_gen[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        seg = F.grid_sample(smpl_x.seg[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        pairs = torch.stack([torch.LongTensor(smpl_x.face).view(-1), torch.LongTensor(smpl_x.face_uv).view(-1)], dim=1).cuda()
        unique_pairs = pairs.unique(dim=0)
        is_seam = (torch.bincount(unique_pairs[:, 0], minlength=smpl_x.vertex_num)[:,None] > 1).float()
        _, uv, is_seam, rgb_smoothed, rgb_aux_smoothed, seg_smoothed = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num,3)).float().cuda(), [uv, is_seam, rgb, rgb_aux, seg]) # upsample with dummy vertex
        is_seam = (is_seam > 0).float()

        # rgb
        rgb = F.grid_sample(smpl_x.texture[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        rgb = rgb*(1-is_seam) + rgb_smoothed*is_seam
        rgb = torch.logit(rgb, eps=1e-4)
        self.rgb = nn.Parameter(rgb.clone())
        self.rgb_orig = rgb.clone()

        rgb_aux = F.grid_sample(smpl_x.texture_gen[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        rgb_aux = rgb_aux*(1-is_seam) + rgb_aux_smoothed*is_seam
        rgb_aux = torch.logit(rgb_aux, eps=1e-4)
        self.rgb_aux = nn.Parameter(rgb_aux.clone())

        # part segmentation
        seg = F.grid_sample(smpl_x.seg[None], uv[None,:,None,:]*2-1, align_corners=True)[0,:,:,0].permute(1,0)
        seg = seg*(1-is_seam) + seg_smoothed*is_seam
        seg = torch.logit(seg, eps=1e-4)
        self.seg = nn.Parameter(seg)

    def get_optimizable_params(self):
        optimizable_params = [
            {'params': [self.mean_offset], 'name': 'mean_offset', 'lr': cfg.lr},
            {'params': [self.scale], 'name': 'scale', 'lr': cfg.lr*10},
            {'params': [self.rgb], 'name': 'rgb', 'lr': cfg.lr*10},
            {'params': [self.rgb_aux], 'name': 'rgb_aux', 'lr': cfg.lr*10},
            {'params': [self.seg], 'name': 'seg', 'lr': cfg.lr*10},
            {'params': [self.triplane], 'name': 'triplane', 'lr': cfg.lr},
            {'params': list(self.mean_offset_offset_net.parameters()), 'name': 'mean_offset_offset_net', 'lr': cfg.lr}
        ]
        return optimizable_params

    def get_neutral_pose_human(self, jaw_zero_pose, use_id_info, return_joint=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_body_pose = smpl_x.neutral_body_pose.view(1,-1).cuda() # 大 pose
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint['part_idx']['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1,3).cuda() # open mouth
        if use_id_info:
            shape_param = self.shape_param[None,:]
            face_offset = self.face_offset[None,:,:]
            joint_offset = self.joint_offset[None,:,:]
        else:
            shape_param = torch.zeros((1,smpl_x.shape_param_dim)).float().cuda()
            face_offset = None
            joint_offset = None
        output = self.smplx_layer(global_orient=zero_pose, body_pose=neutral_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        vert_neutral_pose = output.vertices[0] # 大 pose human
        vert_neutral_pose_upsampled = smpl_x.upsample_mesh(vert_neutral_pose) # 大 pose human
        joint_neutral_pose = output.joints[0][:smpl_x.joint['num'],:] # 大 pose human
        if not return_joint:
            return vert_neutral_pose_upsampled, vert_neutral_pose
        else:
            return vert_neutral_pose_upsampled, vert_neutral_pose, joint_neutral_pose

    def get_zero_pose_human(self, return_vert=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        zero_body_pose = torch.zeros((1,(len(smpl_x.joint['part_idx']['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint['part_idx']['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        shape_param = self.shape_param[None,:]
        face_offset = self.face_offset[None,:,:].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        joint_zero_pose = output.joints[0][:smpl_x.joint['num'],:] # zero pose human
        if not return_vert:
            return joint_zero_pose
        else: 
            vert_zero_pose = output.vertices[0] # zero pose human
            vert_zero_pose_upsampled = smpl_x.upsample_mesh(vert_zero_pose) # zero pose human
            return vert_zero_pose_upsampled, vert_zero_pose, joint_zero_pose
    
    def get_transform_mat_joint(self, joint_zero_pose, smplx_param, jaw_zero_pose):
        # 1. 大 pose -> zero pose
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_body_pose = smpl_x.neutral_body_pose.view(len(smpl_x.joint['part_idx']['body'])-1,3).cuda() # 大 pose
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1,3).cuda() # open mouth
        zero_hand_pose = torch.zeros((len(smpl_x.joint['part_idx']['lhand']),3)).float().cuda()
        pose = torch.cat((zero_pose, neutral_body_pose, jaw_pose, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose)) # follow smpl_x.joint['name']
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_1 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_joint_1 = torch.inverse(transform_mat_joint_1[0])

        # 2. zero pose -> image pose
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint['part_idx']['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint['part_idx']['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint['part_idx']['rhand']),3)
        trans = smplx_param['trans'].view(1,3)
        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) # follow smpl_x.joint['name']
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_joint_2 = transform_mat_joint_2[0]
        translation_mat = torch.zeros((smpl_x.joint['num'],4,4)).float().cuda()
        translation_mat[:,:3,3] = trans # global translation
        transform_mat_joint_2 += translation_mat
        
        # 3. combine 1. 大 pose -> zero pose and 2. zero pose -> image pose
        transform_mat_joint = torch.bmm(transform_mat_joint_2, transform_mat_joint_1)
        return transform_mat_joint
    
    def get_skinning_weight(self, mean_3d, scale):
        # offsets to consider isotropic Gaussian scales (+-2 sigma)
        # weights based on Gaussian PDF values: 
        # +-1 sigma ≈ 0.6065 (~0.5 for simplicity), +-2 sigma ≈ 0.1353 (~0.3 for simplicity)
        offsets = torch.FloatTensor([
            [0,0,0],
            [1,0,0],[-1,0,0],
            [0,1,0],[0,-1,0],
            [0,0,1],[0,0,-1],
            [2,0,0],[-2,0,0],
            [0,2,0],[0,-2,0],
            [0,0,2],[0,0,-2],
        ]).cuda() # (offset_num, 3)
        weights = torch.FloatTensor([1.0] + [0.5 for _ in range(6)] + [0.3 for _ in range(6)]).cuda() # (offset_num,)
        offset_num = offsets.shape[0]

        # prepare sampling coordinates by normalizing them to [-1,1]
        def normalize_coords(points, x, y, z):
            nx = 2 * (points[:, 0] - x.min()) / (x.max() - x.min()) - 1
            ny = 2 * (points[:, 1] - y.min()) / (y.max() - y.min()) - 1
            nz = 2 * (points[:, 2] - z.min()) / (z.max() - z.min()) - 1
            return torch.stack([nx, ny, nz], dim=1) 
        xyz = mean_3d[:,None,:] + offsets[None,:,:] * scale[:,None,:] # (smpl_x.vertex_num_upsampled, offset_num, 3)
        xyz = xyz.view(-1,3)
        xyz = normalize_coords(xyz, self.skinning_weight_grid[:,0], self.skinning_weight_grid[:,1], self.skinning_weight_grid[:,2])

        # trilinear interpolation from diffused skinning weight field
        grid = xyz.view(1, smpl_x.vertex_num_upsampled*offset_num, 1, 1, 3)
        field = self.skinning_weight.permute(3,2,1,0)[None] # (1, smpl_x.joint['num'], D_z, D_y, D_x)
        skinning_weight = F.grid_sample(field, grid, mode='bilinear', align_corners=True) # (1, smpl_x.joint['num'], smpl_x.vertex_num_upsampled*offset_num, 1, 1)
        skinning_weight = skinning_weight[0,:,:,0,0].permute(1,0).reshape(smpl_x.vertex_num_upsampled, offset_num, smpl_x.joint['num'])
        skinning_weight = torch.clamp(skinning_weight, min=0)

        # weighted sum the sampled skinning weight
        skinning_weight = (skinning_weight * weights[None,:,None]).sum(dim=1) / (weights.sum() + 1e-8) # (smpl_x.vertex_num_upsampled, smpl_x.joint['num'])
        skinning_weight = F.normalize(torch.clamp(skinning_weight, min=0), p=1, dim=1)

        # for hands and face, assign original vertex index to use skinning weight of the original vertex
        mask = ((self.is_hand + self.is_face) > 0).float()[:,None] 
        skinning_weight = self.skinning_weight_orig*mask + skinning_weight*(1-mask)
        return skinning_weight

    def lbs(self, xyz, transform_mat_vertex):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:1])),1) # 大 pose. xyz1
        xyz = torch.bmm(transform_mat_vertex, xyz[:,:,None]).view(smpl_x.vertex_num_upsampled,4)[:,:3]
        return xyz

    def extract_tri_feature(self):
        # normalize coordinates to [-1,1]
        xyz = self.pos_enc_vert
        xyz = xyz - torch.mean(xyz,0)[None,:]
        x = xyz[:,0] / (cfg.triplane_shape_3d[0]/2)
        y = xyz[:,1] / (cfg.triplane_shape_3d[1]/2)
        z = xyz[:,2] / (cfg.triplane_shape_3d[2]/2)

        # extract features from the triplane
        xy, xz, yz = torch.stack((x,y),1), torch.stack((x,z),1), torch.stack((y,z),1)
        feat_xy = F.grid_sample(self.triplane[0,None,:,:,:], xy[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_xz = F.grid_sample(self.triplane[1,None,:,:,:], xz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        feat_yz = F.grid_sample(self.triplane[2,None,:,:,:], yz[None,:,None,:])[0,:,:,0] # cfg.triplane_shape[0], smpl_x.vertex_num_upsampled
        tri_feat = torch.cat((feat_xy, feat_xz, feat_yz)).permute(1,0) # smpl_x.vertex_num_upsampled, cfg.triplane_shape[0]*3
        return tri_feat

    def get_mean_offset_offset(self, tri_feat, smplx_param, joint_idxs):
        # pose from smplx parameters (only use body pose as face/hand poses are not diverse in the training set)
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint['part_idx']['body'])-1,3)

        # combine pose features with triplane feature
        # for pose, take 4-ring joints for each vertex for better generalizability
        adj_joint_mask = smpl_x.adj_joint_mask.cuda()[joint_idxs,:] # smpl_x.vertex_num_upsampled, smpl_x.joint['num']
        adj_joint_mask = adj_joint_mask[:,smpl_x.joint['part_idx']['body']][:,1:] # take only body joint mask and exclude the root joint.
        pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).view(1,len(smpl_x.joint['part_idx']['body'])-1,6).repeat(smpl_x.vertex_num_upsampled,1,1) # without root pose
        pose = (pose * adj_joint_mask[:,:,None]).view(smpl_x.vertex_num_upsampled, (len(smpl_x.joint['part_idx']['body'])-1)*6) # for joints that are not included in the 4-ring, make poses zero
     
        # forward to geometry networks
        feat = torch.cat((tri_feat, pose.detach()),1)
        mean_offset_offset = self.mean_offset_offset_net(feat)/100 # pose-dependent mean offset of Gaussians

        # for hands, eyes, and face, do not use offsets as they have very small pose-dependent deformations
        mask = ((self.is_hand + self.is_face_expr + self.is_eye) > 0)[:,None].float()
        mean_offset_offset = mean_offset_offset * (1 - mask)
        return mean_offset_offset

    def forward(self, smplx_param, cam_param):
        vert_neutral_pose, vert_neutral_pose_wo_upsample = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        joint_zero_pose = self.get_zero_pose_human()
        
        # get geometry Gaussian features
        mean_offset = self.mean_offset
        scale = torch.exp(self.scale).repeat(1,3)
        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # constant
        opacity = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda() # constant
        rgb = torch.sigmoid(self.rgb)
        mean_3d = vert_neutral_pose + mean_offset # 大 pose

        # get skinning weight
        skinning_weight = self.get_skinning_weight(mean_3d, scale.detach())

        # pose-dependent mean offsets
        tri_feat = self.extract_tri_feature()
        joint_idxs = torch.argmax(skinning_weight,1)
        mean_offset_offset = self.get_mean_offset_offset(tri_feat, smplx_param, joint_idxs)
        mean_3d_refined = mean_3d + mean_offset_offset # 大 pose

        # smplx facial expression offset
        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.expr_dirs).sum(2)
        vert = vert_neutral_pose + smplx_expr_offset # 大 pose
        mean_3d = mean_3d + smplx_expr_offset # 大 pose
        mean_3d_refined = mean_3d_refined + smplx_expr_offset # 大 pose

        # get skinning weight
        skinning_weight_refined = self.get_skinning_weight(mean_3d_refined, scale.detach())

        # forward kinematics and lbs
        transform_mat_joint = self.get_transform_mat_joint(joint_zero_pose, smplx_param, jaw_zero_pose=True) # follow jaw_pose of the vert_neutral_pose
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(smpl_x.joint['num'],16)).view(smpl_x.vertex_num_upsampled,4,4)
        vert = self.lbs(vert, transform_mat_vertex) # posed with smplx_param
        mean_3d = self.lbs(mean_3d, transform_mat_vertex) # posed with smplx_param
        transform_mat_vertex = torch.matmul(skinning_weight_refined, transform_mat_joint.view(smpl_x.joint['num'],16)).view(smpl_x.vertex_num_upsampled,4,4)
        mean_3d_refined = self.lbs(mean_3d_refined, transform_mat_vertex) # posed with smplx_param

        # camera coordinate system -> world coordinate system
        vert = torch.matmul(torch.inverse(cam_param['R']), (vert - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        mean_3d = torch.matmul(torch.inverse(cam_param['R']), (mean_3d - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        mean_3d_refined = torch.matmul(torch.inverse(cam_param['R']), (mean_3d_refined - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)

        # Gaussians and offsets
        assets = {'mean_3d': mean_3d, 'opacity': opacity, 'scale': scale, 'rotation': rotation, 'rgb': rgb}
        assets_refined = {'mean_3d': mean_3d_refined, 'opacity': opacity, 'scale': scale, 'rotation': rotation, 'rgb': rgb}
        offsets = {'mean_offset': mean_offset, 'mean_offset_offset': mean_offset_offset}
        return assets, assets_refined, offsets, vert_neutral_pose, vert

class GaussianRenderer(nn.Module):
    def __init__(self):
        super(GaussianRenderer, self).__init__()
    
    def forward(self, gaussian_assets, img_shape, cam_param, bg=torch.ones((3)).float().cuda()):
        # assets for the rendering
        mean_3d = gaussian_assets['mean_3d']
        opacity = gaussian_assets['opacity']
        scale = gaussian_assets['scale']
        rotation = gaussian_assets['rotation']
        rgb = gaussian_assets['rgb']

        # create rasterizer
        # permute view_matrix and proj_matrix following GaussianRasterizer's configuration following below links
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L54
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/cameras.py#L55
        fov = get_fov(cam_param['focal'], img_shape)
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3,:3]
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            kernel_size=0.1,
            subpixel_offset=torch.zeros((img_shape[0],img_shape[1],2)).float().cuda(),
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=view_matrix, 
            projmatrix=full_proj_matrix,
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num,3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()
        
        # rasterize visible Gaussians to image and obtain their radius (on screen)
        render_img, radius = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scale,
            rotations=rotation,
            cov3D_precomp=None
        )
        
        out = {'img': render_img, 'mean_2d': mean_2d, 'is_vis': radius > 0, 'radius': radius}
        return out

class SMPLXParamDict(nn.Module):
    def __init__(self):
        super(SMPLXParamDict, self).__init__()

    # initialize SMPL-X parameters of all frames
    def init(self, smplx_params):
        _smplx_params = {}
        for split in smplx_params.keys():
            _smplx_params[str(split)] = nn.ParameterDict({})
            for frame_idx in smplx_params[split].keys():
                _smplx_params[str(split)][str(frame_idx)] = nn.ParameterDict({})
                for param_name in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                    if 'pose' in param_name:
                        _smplx_params[str(split)][str(frame_idx)][param_name] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(smplx_params[split][frame_idx][param_name].cuda())))
                    else:
                        _smplx_params[str(split)][str(frame_idx)][param_name] = nn.Parameter(smplx_params[split][frame_idx][param_name].cuda())

        self.smplx_params = nn.ParameterDict(_smplx_params)

    def get_optimizable_params(self):
        optimizable_params = []
        for split in self.smplx_params.keys():
            for frame_idx in self.smplx_params[split].keys():
                for param_name in self.smplx_params[split][frame_idx].keys():
                    if 'hand_pose' in param_name:
                        lr = cfg.smplx_param_lr / 10
                    else:
                        lr = cfg.smplx_param_lr
                    
                    optimizable_params.append({'params': [self.smplx_params[split][frame_idx][param_name]], 'name': 'smplx_param_' + param_name + '_' + split + '_' + frame_idx, 'lr': lr})
        return optimizable_params
    
    def forward(self, splits, frame_idxs):
        out = []
        for split, frame_idx in zip(splits,frame_idxs):
            split = str(split)
            frame_idx = str(int(frame_idx))
            smplx_param = {}
            for param_name in self.smplx_params[split][frame_idx].keys():
                if 'pose' in param_name:
                    smplx_param[param_name] = matrix_to_axis_angle(rotation_6d_to_matrix(self.smplx_params[split][frame_idx][param_name]))
                else:
                    smplx_param[param_name] = self.smplx_params[split][frame_idx][param_name]
            out.append(smplx_param)
        return out
