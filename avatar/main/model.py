import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.module import PERSONA, SMPLXParamDict, GaussianRenderer
from nets.layer import MeshRenderer, rasterize
from nets.loss import ImgLoss, GeoLoss, LaplacianReg, TVReg
from utils.smpl_x import smpl_x
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from kornia.morphology import dilation
from kornia.filters import spatial_gradient
from smplx.lbs import batch_rigid_transform
import copy
from config import cfg

class Model(nn.Module):
    def __init__(self, persona, smplx_param_dict):
        super(Model, self).__init__()
        self.persona = persona
        self.smplx_param_dict = smplx_param_dict
        self.gaussian_renderer = GaussianRenderer()
        self.mesh_renderer = MeshRenderer()
        self.smplx_layer = copy.deepcopy(smpl_x.layer)

        self.img_loss = ImgLoss()
        self.geo_loss = GeoLoss()
        self.lap_reg = LaplacianReg(smpl_x.vertex_num_upsampled, smpl_x.face_upsampled)
        self.lap_reg_lr = LaplacianReg(smpl_x.vertex_num, smpl_x.face)
        self.tv_reg = TVReg()

        if cfg.fit_pose_to_test:
            self.optimizable_params = self.smplx_param_dict.get_optimizable_params()
            self.eval_modules = [self.img_loss]
        else:
            self.optimizable_params = self.persona.get_optimizable_params() 
            if smplx_param_dict is not None:
                self.optimizable_params += self.smplx_param_dict.get_optimizable_params()
            self.eval_modules = [self.img_loss]

    def get_smplx_outputs(self, smplx_param, cam_param):
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(1,(len(smpl_x.joint['part_idx']['body'])-1)*3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(1,len(smpl_x.joint['part_idx']['lhand'])*3)
        rhand_pose = smplx_param['rhand_pose'].view(1,len(smpl_x.joint['part_idx']['rhand'])*3)
        expr = smplx_param['expr'].view(1,smpl_x.expr_param_dim)
        trans = smplx_param['trans'].view(1,3)

        shape = self.persona.shape_param[None]
        face_offset = self.persona.face_offset[None].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.persona.joint_offset[None])

        # camera coordinate system
        output = self.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
        vert, kpt = output.vertices[0], output.joints[0,smpl_x.kpt['idx'],:]

        # camera coordinate system -> world coordinate system
        vert = torch.matmul(torch.inverse(cam_param['R']), (vert - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        kpt = torch.matmul(torch.inverse(cam_param['R']), (kpt - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        return vert, kpt
    
    def render_mask(self, asset, cam_param, render_shape):
        asset_mask = {k: torch.ones_like(v) if k == 'rgb' else v for k,v in asset.items()}
        mask = self.gaussian_renderer(asset_mask, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img'][0,None,:,:]
        return mask
    
    def render_img_aux(self, asset, cam_param, render_shape, bg, mask):
        asset_img_aux = {k: torch.sigmoid(self.persona.rgb_aux) if k == 'rgb' else v.detach() for k,v in asset.items()} # color: rgb_aux, geo: detach
        img_aux = self.gaussian_renderer(asset_img_aux, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img']
        img_aux = img_aux + bg*(1-mask).detach()
        return img_aux

    def render_img_geo_detach(self, asset, cam_param, render_shape, bg, mask):
        asset_geo_detach = {k: v if k == 'rgb' else v.detach() for k,v in asset.items()}
        img_geo_detach = self.gaussian_renderer(asset_geo_detach, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img']
        img_geo_detach = img_geo_detach + bg*(1-mask).detach()
        return img_geo_detach

    def render_hand_img(self, asset, cam_param, render_shape):
        hand_img = []
        for hand_type in ('right', 'left'):
            if hand_type == 'right':
                is_hand = self.persona.is_rhand
            else:
                is_hand = self.persona.is_lhand
            asset_hand = {k: v[is_hand,:] for k,v in asset.items()}
            asset_hand = {k: v if k == 'rgb' else v.detach() for k,v in asset_hand.items()} # detach geometry
            hand_img.append(self.gaussian_renderer(asset_hand, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img'])
        hand_img = torch.cat(hand_img)
        return hand_img

    def render_hand_mask(self, asset, cam_param, render_shape):
        hand_mask = []
        for hand_type in ('right', 'left'):
            if hand_type == 'right':
                is_hand = self.persona.is_rhand
            else:
                is_hand = self.persona.is_lhand
            asset_hand = {k: v[is_hand,:] for k,v in asset.items()}
            asset_hand = {k: torch.ones_like(v) if k == 'rgb' else v for k,v in asset_hand.items()} 
            hand_mask.append(self.gaussian_renderer(asset_hand, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img'][0][None])
        hand_mask = torch.cat(hand_mask)
        return hand_mask
    
    def render_face_img(self, asset, cam_param, render_shape, bg, mask, rgb_mode=None):
        is_face = ((self.persona.is_face_expr + self.persona.is_eye) > 0).float()[:,None]
        if rgb_mode is None:
            asset_img = {k: v*is_face if k == 'rgb' else v for k,v in asset.items()} # black background color for others
        elif rgb_mode == 'init':
            asset_img = {k: torch.sigmoid(self.persona.rgb_orig)*is_face if k == 'rgb' else v for k,v in asset.items()} # constant initial rgb for the face black background color for others
        elif rgb_mode == 'aux':
            asset_img = {k: torch.sigmoid(self.persona.rgb_aux)*is_face if k == 'rgb' else v for k,v in asset.items()} # auxiliary rgb for the face black background color for others
        img_face = self.gaussian_renderer(asset_img, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img']
        img_face = img_face + bg*(1-mask)
        return img_face
    
    def render_face_mask(self, asset, cam_param, render_shape):
        is_face = ((self.persona.is_face_expr + self.persona.is_eye) > 0)
        asset_mask = {k: torch.zeros_like(v) if k == 'rgb' else v for k,v in asset.items()}
        asset_mask['rgb'][is_face,:] = 1
        mask_face = self.gaussian_renderer(asset_mask, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img'][0,None,:,:]
        return mask_face

    def render_eye_mask(self, asset, cam_param, render_shape):
        asset_mask = {k: torch.zeros_like(v) if k == 'rgb' else v for k,v in asset.items()}
        asset_mask['rgb'][self.persona.is_eye,:] = 1
        mask_eye = self.gaussian_renderer(asset_mask, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img'][0,None,:,:]
        return mask_eye

    def render_depth(self, asset, cam_param, render_shape):
        xyz = torch.matmul(cam_param['R'], asset['mean_3d'].permute(1,0)).permute(1,0) + cam_param['t'].view(1,3) # world coordinate -> camera coordinate
        asset_depth = {k: xyz[:,2,None].repeat(1,3) if k == 'rgb' else v.detach() for k,v in asset.items()}
        depth = self.gaussian_renderer(asset_depth, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img']
        depth = depth[0,None,:,:]
        return depth

    def render_normal(self, asset, cam_param, render_shape):
        xyz = torch.matmul(cam_param['R'], asset['mean_3d'].permute(1,0)).permute(1,0) + cam_param['t'].view(1,3) # world coordinate -> camera coordinate
        normal = Meshes(verts=xyz[None], faces=torch.LongTensor(smpl_x.face_upsampled).cuda()[None]).verts_normals_packed().reshape(smpl_x.vertex_num_upsampled,3)
        normal = torch.stack((normal[:,0], -normal[:,1], -normal[:,2]),1) # flip y-axis and z-axis to match the convention
        asset_normal = {k: normal if k == 'rgb' else v for k,v in asset.items()}
        normal = self.gaussian_renderer(asset_normal, render_shape, cam_param, bg=torch.ones((3)).float().cuda()*-1)['img']
        normal = (normal + 1)/2. # [0,1]
        return normal
    
    def render_seg(self, asset, cam_param, render_shape):
        asset_seg = {k: torch.sigmoid(self.persona.seg) if k == 'rgb' else v for k,v in asset.items()}
        seg = self.gaussian_renderer(asset_seg, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img']
        return seg

    def render_hand_mesh(self, xyz, cam_param, render_shape):
        # rasterize mesh
        fragments_rhand = rasterize(xyz[None,smpl_x.rhand_vertex_idx,:], smpl_x.rhand_face, {k: v[None] for k,v in cam_param.items()}, render_shape)
        fragments_lhand = rasterize(xyz[None,smpl_x.lhand_vertex_idx,:], smpl_x.lhand_face, {k: v[None] for k,v in cam_param.items()}, render_shape)

        # get foreground mask
        mask_rhand = (fragments_rhand.pix_to_face[0,:,:,0] != -1)
        mask_lhand = (fragments_lhand.pix_to_face[0,:,:,0] != -1)
        mask_hand = torch.stack((mask_rhand, mask_lhand)).float()
        return mask_hand
    
    def render_face_mesh(self, xyz, cam_param, render_shape, add_mouth_in=False):
        # get face mask
        is_face = torch.zeros((smpl_x.vertex_num)).float().cuda()
        is_face[smpl_x.layer.lbs_weights.argmax(1)==smpl_x.joint['name'].index('R_Eye')] = 1
        is_face[smpl_x.layer.lbs_weights.argmax(1)==smpl_x.joint['name'].index('L_Eye')] = 1
        is_face[torch.from_numpy(smpl_x.expr_vertex_idx).long().cuda()] = 1
        is_face = is_face[None,:]

        # downsample mesh
        vertex_idx = smpl_x.lr_idx_to_hr_idx(torch.arange(smpl_x.vertex_num))
        xyz = xyz[None,vertex_idx,:]

        # render
        cam_param = {k: v[None] for k,v in cam_param.items()}
        if not add_mouth_in:
            mask_face = self.mesh_renderer(is_face, xyz, smpl_x.face, cam_param, render_shape)[0]
        else:
            mask_face = self.mesh_renderer(is_face, xyz, smpl_x.face_with_mouth_in, cam_param, render_shape)[0]
        return mask_face

    def render_eye_mesh(self, xyz, cam_param, render_shape):
        # get eye mask
        is_eye = torch.zeros((smpl_x.vertex_num)).float().cuda()
        is_eye[smpl_x.layer.lbs_weights.argmax(1)==smpl_x.joint['name'].index('R_Eye')] = 1
        is_eye[smpl_x.layer.lbs_weights.argmax(1)==smpl_x.joint['name'].index('L_Eye')] = 1
        is_eye = is_eye[None,:]

        # downsample mesh
        vertex_idx = smpl_x.lr_idx_to_hr_idx(torch.arange(smpl_x.vertex_num))
        xyz = xyz[None,vertex_idx,:]

        # render
        cam_param = {k: v[None] for k,v in cam_param.items()}
        mask_eye = self.mesh_renderer(is_eye, xyz, smpl_x.face, cam_param, render_shape)[0]
        return mask_eye

    def get_boundary(self, asset, pos_enc, cam_param, render_shape, mask, mask_face, vertex_mask=None, kernel_ratio=cfg.img_kernel_ratio):
        
        # Gaussian positions in the template space
        pos_enc = pos_enc - torch.mean(pos_enc,0)[None,:]
       
        # render positional encoding map
        r = (pos_enc[:,0] / (cfg.triplane_shape_3d[0]/2) + 1)/2 # [0,1]
        g = (pos_enc[:,1] / (cfg.triplane_shape_3d[1]/2) + 1)/2 # [0,1]
        b = (pos_enc[:,2] / (cfg.triplane_shape_3d[2]/2) + 1)/2 # [0,1]
        rgb = torch.stack((r,g,b),1)
        if vertex_mask is not None:
            asset_render = {k: rgb*vertex_mask if k == 'rgb' else v for k,v in asset.items()}
        else:
            asset_render = {k: rgb if k == 'rgb' else v for k,v in asset.items()}
        pos_enc_render = self.gaussian_renderer(asset_render, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())['img'][None,:,:,:]

        # apply sobel filter to get boundary
        grad = spatial_gradient(pos_enc_render, mode='sobel', order=1) # (1, 3, 2, height, width)
        grad_mag = torch.norm(grad, dim=2).mean(dim=1)[0] # (height, width)
        boundary = (grad_mag > cfg.boundary_thr).float()[None,:,:] # (1, height, width)
        
        # ignore face inside and dilate boundary
        boundary = boundary * (1 - mask_face)
        if (mask > 0.9).sum() > 0:
            kernel_size = self.get_kernel_size((mask>0.9).float(), kernel_ratio)
            kernel = torch.ones((kernel_size,kernel_size)).float().cuda()
            boundary = dilation(boundary[None], kernel)[0]
        return boundary
    
    def get_kernel_size(self, mask, ratio):
        if mask.ndim == 3:
            mask = mask[0]
        ys, xs = torch.where(mask > 0)
        size = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        kernel_size = int(round(float(ratio * size)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def forward(self, data, mode):
        batch_size = data['img'].shape[0]
        
        # background color for renderings
        bg_black = torch.zeros((3)).float().cuda()
        bg_white = torch.ones((3)).float().cuda()
        
        # placeholder 
        smplx_verts = {'lr': [], 'hr': []}
        assets = {'mean_3d': [], 'scale': [], 'rotation': [], 'opacity': [], 'rgb': []}
        assets_refined = {'mean_3d': [], 'scale': [], 'rotation': [], 'opacity': [], 'rgb': []}
        offsets = {'mean_offset': [], 'mean_offset_offset': []}
        renders = {'img': [], 'img_refined': [], 'mask': [], 'mask_refined': [], 'depth': [], 'depth_refined': [], 'normal': [], 'normal_refined': []}
        if mode == 'train':
            renders['img_w_bg'], renders['img_refined_w_bg'], renders['img_w_albedo_bg'], renders['img_aux_w_bg'], renders['img_aux_refined_w_bg'], renders['img_w_bg_geo_detach'], renders['img_refined_w_bg_geo_detach'] = [], [], [], [], [], [], []
            renders['seg'], renders['seg_refined'] = [], []
            renders['img_hand'], renders['mask_hand'] = [], []
            renders['mask_eye'], renders['img_face_init'], renders['mask_patch_face'], renders['patch_face'], renders['patch_face_aux'], renders['patch_face_geo_detach'], renders['patch_face_init'] = [], [], [], [], [], [], []
            renders['boundary'], renders['boundary_refined'], renders['boundary_face'] = [], [], []
            mesh_renders = {'mask_hand': [], 'mask_patch_face': [], 'mask_eye': []}
       
        # forward
        for i in range(batch_size):
            cam_param = {k: v[i] for k,v in data['cam_param'].items()}
            cam_param_face = {k: v[i] for k,v in data['cam_param_face'].items()}
            img_height, img_width = data['img'][i].shape[1:]
            render_shape = (img_height, img_width)
            
            # get assets and offsets
            smplx_param = self.smplx_param_dict([data['split'][i]], [data['frame_idx'][i]])[0]
            asset, asset_refined, offset, vert_neutral_pose, vert = self.persona(smplx_param, cam_param)

            # gather assets and offsets
            for key in ['mean_3d', 'scale', 'rotation', 'opacity', 'rgb']:
                assets[key].append(asset[key])
            for key in ['mean_3d', 'scale', 'rotation', 'opacity', 'rgb']:
                assets_refined[key].append(asset_refined[key])
            for key in ['mean_offset', 'mean_offset_offset']:
                offsets[key].append(offset[key])

            # smplx outputs
            smplx_vert, smplx_kpt = self.get_smplx_outputs(smplx_param, cam_param)
            smplx_verts['lr'].append(smplx_vert); smplx_verts['hr'].append(vert);
            
            # image and mask
            renders['mask'].append(self.render_mask(asset, cam_param, render_shape))
            renders['mask_refined'].append(self.render_mask(asset_refined, cam_param, render_shape))
            img_render = self.gaussian_renderer(asset, render_shape, cam_param, bg=bg_black)['img']
            img_render_refined = self.gaussian_renderer(asset_refined, render_shape, cam_param, bg=bg_black)['img']
            renders['img'].append(img_render + bg_white[:,None,None]*(1-renders['mask'][-1]))
            renders['img_refined'].append(img_render_refined + bg_white[:,None,None]*(1-renders['mask_refined'][-1]))
            
            # depths and normal maps
            renders['depth'].append(self.render_depth(asset, cam_param, render_shape))
            renders['depth_refined'].append(self.render_depth(asset_refined, cam_param, render_shape))
            renders['normal'].append(self.render_normal(asset, cam_param, render_shape))
            renders['normal_refined'].append(self.render_normal(asset_refined, cam_param, render_shape))
            
            if mode == 'train':
                # image
                renders['img_w_bg'].append(img_render + data['img'][i]*(1-renders['mask'][-1]))
                renders['img_refined_w_bg'].append(img_render_refined + data['img'][i]*(1-renders['mask_refined'][-1]))
                renders['img_w_albedo_bg'].append(img_render + data['albedo'][i]*(1-renders['mask'][-1]))
                renders['img_aux_w_bg'].append(self.render_img_aux(asset, cam_param, render_shape, data['img'][i], renders['mask'][-1]))
                renders['img_aux_refined_w_bg'].append(self.render_img_aux(asset_refined, cam_param, render_shape, data['img'][i], renders['mask_refined'][-1]))
                renders['img_w_bg_geo_detach'].append(self.render_img_geo_detach(asset, cam_param, render_shape, data['img'][i], renders['mask'][-1]))
                renders['img_refined_w_bg_geo_detach'].append(self.render_img_geo_detach(asset_refined, cam_param, render_shape, data['img'][i], renders['mask_refined'][-1]))
                
                # part segmentations
                renders['seg'].append(self.render_seg(asset, cam_param, render_shape))
                renders['seg_refined'].append(self.render_seg(asset_refined, cam_param, render_shape))
                
                # hand
                renders['img_hand'].append(self.render_hand_img(asset, cam_param, render_shape))
                renders['mask_hand'].append(self.render_hand_mask(asset, cam_param, render_shape))
                with torch.no_grad():
                    mesh_renders['mask_hand'].append(self.render_hand_mesh(smplx_vert, cam_param, render_shape).detach())
                
                # face and eye
                mask_face = self.render_face_mask(asset, cam_param, render_shape)
                renders['img_face_init'].append(self.render_face_img(asset, cam_param, render_shape, data['img'][i], mask_face, 'init'))
                renders['mask_eye'].append(self.render_eye_mask(asset, cam_param, render_shape))
                with torch.no_grad():
                    mesh_renders['mask_eye'].append(self.render_eye_mesh(smplx_vert, cam_param, render_shape).detach())

                # face patch
                renders['mask_patch_face'].append(self.render_face_mask(asset, cam_param_face, cfg.face_patch_shape))
                renders['patch_face'].append(self.render_face_img(asset, cam_param_face, cfg.face_patch_shape, data['face_patch'][i], renders['mask_patch_face'][-1]))
                renders['patch_face_aux'].append(self.render_face_img(asset, cam_param_face, cfg.face_patch_shape, data['face_patch'][i], renders['mask_patch_face'][-1], 'aux'))
                renders['patch_face_geo_detach'].append(self.render_img_geo_detach(asset, cam_param_face, cfg.face_patch_shape, data['face_patch'][i], renders['mask_patch_face'][-1]))
                renders['patch_face_init'].append(self.render_face_img(asset, cam_param_face, cfg.face_patch_shape, data['face_patch'][i], renders['mask_patch_face'][-1], 'init'))
                with torch.no_grad():
                    mesh_renders['mask_patch_face'].append(self.render_face_mesh(smplx_vert, cam_param_face, cfg.face_patch_shape).detach())

                # boundary
                with torch.no_grad():
                    mask_face = (self.render_face_mesh(smplx_vert, cam_param, render_shape, add_mouth_in=True) > 0.9).float()
                    renders['boundary'].append(self.get_boundary(asset, vert_neutral_pose+offset['mean_offset'], cam_param, render_shape, renders['mask'][-1], mask_face).detach())
                    renders['boundary_refined'].append(self.get_boundary(asset_refined, vert_neutral_pose+offset['mean_offset']+offset['mean_offset_offset'], cam_param, render_shape, renders['mask_refined'][-1], mask_face).detach())
                    is_face = ((self.persona.is_eye + self.persona.is_face_expr) > 0).float()[:,None]
                    mask_patch_face = (self.render_face_mesh(smplx_vert, cam_param_face, cfg.face_patch_shape, add_mouth_in=True) > 0.9).float()
                    renders['boundary_face'].append(self.get_boundary(asset, vert_neutral_pose+offset['mean_offset'], cam_param_face, cfg.face_patch_shape, renders['mask_patch_face'][-1], mask_patch_face, is_face, cfg.patch_kernel_ratio).detach())
                
                """
                # for debug
                import cv2
                import os
                import os.path as osp
                is_captured = float(data['split'][0] == 'captured')
                is_generated = float('generated' in data['split'][0])
                if not cfg.fit_pose_to_test:
                    save_root_path = cfg.subject_id
                else:
                    save_root_path = cfg.subject_id + '_fit_pose_to_test'
                os.makedirs(save_root_path, exist_ok=True)
                prefix = str(int(data['frame_idx'][0])) + '_' + str(data['split'][0][:3])
                if is_generated:
                    prefix += data['split'][0].split('_')[-1]
                cv2.imwrite(osp.join(save_root_path, prefix + '_orig_a.jpg'), data['img'][0].detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
                cv2.imwrite(osp.join(save_root_path, prefix + '_img_a.jpg'), (renders['img'][0]+1-renders['mask'][0]).detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
                cv2.imwrite(osp.join(save_root_path, prefix + '_img_face_init_a.jpg'), (renders['img_face_init'][0]+1-renders['mask'][0]).detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
                #cv2.imwrite(osp.join(save_root_path, prefix + '_normal_a.jpg'), renders['normal'][0].detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
                #cv2.imwrite(osp.join(save_root_path, prefix + '_boundary_a.jpg'), renders['boundary'][0].detach().cpu().numpy().transpose(1,2,0)*255)
                #cv2.imwrite(osp.join(save_root_path, prefix + '_boundary_face_a.jpg'), renders['boundary_face'][0].detach().cpu().numpy().transpose(1,2,0)*255)
                cv2.imwrite(osp.join(save_root_path, prefix + '_img_refined_a.jpg'), (renders['img_refined'][0]+1-renders['mask_refined'][0]).detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
                #cv2.imwrite(osp.join(save_root_path, prefix + '_normal_refined_a.jpg'), renders['normal_refined'][0].detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
                #cv2.imwrite(osp.join(save_root_path, prefix + '_boundary_refined_a.jpg'), renders['boundary_refined'][0].detach().cpu().numpy().transpose(1,2,0)*255)
                cv2.imwrite(osp.join(save_root_path, prefix + '_face_patch_a.jpg'), (renders['patch_face'][0]*renders['mask_patch_face'][0]+1-renders['mask_patch_face'][0]).detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
                cv2.imwrite(osp.join(save_root_path, prefix + '_face_patch_init_a.jpg'), (renders['patch_face_init'][0]*renders['mask_patch_face'][0]+1-renders['mask_patch_face'][0]).detach().cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
                """
           
        # aggregate assets and renders
        smplx_verts = {k: torch.stack(v) for k,v in smplx_verts.items()}
        assets = {k: torch.stack(v) for k,v in assets.items()}
        assets_refined = {k: torch.stack(v) for k,v in assets_refined.items()}
        offsets = {k: torch.stack(v) for k,v in offsets.items()}
        renders = {k: torch.stack(v) for k,v in renders.items()}
        if mode == 'train':
            mesh_renders = {k: torch.stack(v) for k,v in mesh_renders.items()}
        
        if mode == 'train':
            is_captured = torch.FloatTensor([x=='captured' for x in data['split']]).cuda()[:,None,None,None]
            is_generated = torch.FloatTensor(['generated' in x for x in data['split']]).cuda()[:,None,None,None]
            
            # loss functions
            loss = {}
            
            # image
            if not cfg.fit_pose_to_test:
                weight = is_captured*(renders['boundary']*0.1 + (1-renders['boundary'])) + is_generated
            else:
                weight = 1
            loss['img'] = self.img_loss(renders['img_w_bg'], data['img']) * weight
            if is_captured.sum() > 0:
                loss['img_albedo'] = self.img_loss(renders['img_w_albedo_bg'], data['albedo']) * weight * is_captured
            if not cfg.fit_pose_to_test:
                weight = weight * 0.1
            else:
                weight = 1
            loss['img_refined'] = self.img_loss(renders['img_refined_w_bg'], data['img']) * weight
            if is_generated.sum() > 0:
                loss['img_aux_refined'] = self.img_loss(renders['img_aux_refined_w_bg'], data['img']) * is_generated
            if is_captured.sum() > 0:
                loss['img_boundary'] = self.img_loss(renders['img_w_bg_geo_detach']*renders['boundary'], renders['img_aux_w_bg'].detach()*renders['boundary']) * is_captured
                loss['img_boundary_refined'] = self.img_loss(renders['img_refined_w_bg_geo_detach']*renders['boundary_refined'], renders['img_aux_refined_w_bg'].detach()*renders['boundary_refined']) * is_captured

            # geometry
            if not cfg.fit_pose_to_test:
                weight = is_captured*0.1 + is_generated
            else:
                weight = 1
            loss['geo'] = torch.abs(renders['mask'] - data['mask']) + self.geo_loss(renders['depth'], renders['normal'], renders['seg'], data['depth'], data['normal'], data['seg']) * weight
            loss['geo_refined'] = torch.abs(renders['mask_refined'] - data['mask']) + self.geo_loss(renders['depth_refined'], renders['normal_refined'], renders['seg_refined'], data['depth'], data['normal'], data['seg']) * weight
            if not cfg.fit_pose_to_test:
                loss['mask_hand'] = torch.abs(renders['mask_hand'] - mesh_renders['mask_hand']) * is_generated * 50
                loss['face_geo'] = torch.abs(renders['img_face_init'] - data['img']) * is_captured
                loss['mask_eye'] = torch.abs(renders['mask_eye'] - mesh_renders['mask_eye'])

            if cfg.fit_pose_to_test:
                return loss
            
            # face patch
            if is_captured.sum() > 0:
                weight = renders['boundary_face']*0.1 + (1 - renders['boundary_face'])
                loss['patch_face'] = self.img_loss(renders['patch_face'], data['face_patch']) * weight * is_captured * 0.1
                loss['patch_face_boundary'] = self.img_loss(renders['patch_face_geo_detach']*renders['boundary_face'], renders['patch_face_aux'].detach()*renders['boundary_face']) * is_captured * 0.1
                loss['patch_mask_face'] = torch.abs(renders['mask_patch_face'] - mesh_renders['mask_patch_face']) * is_captured * 0.1
                loss['patch_face_geo'] = torch.abs(renders['patch_face_init'] - data['face_patch']) * is_captured * 0.1
           
            # geometry regularizers
            weight = torch.ones((1,smpl_x.vertex_num_upsampled,1)).float().cuda(); weight[:,self.persona.is_eye+self.persona.is_face_expr,:] = 50; weight[:,self.persona.is_hand+self.persona.is_expr_boundary,:] = 10; weight *= 100000;
            use_template = torch.zeros((1,smpl_x.vertex_num_upsampled,1)).float().cuda(); use_template[:,self.persona.is_hand+self.persona.is_eye+self.persona.is_face_expr,:] = 1
            loss['lap_mean'] = (self.lap_reg(vert_neutral_pose[None]+offsets['mean_offset'], vert_neutral_pose[None]) * use_template + self.lap_reg(vert_neutral_pose[None]+offsets['mean_offset'], None) * (1 - use_template)) * weight * 10
            loss['lap_mean_refined'] = self.lap_reg(assets_refined['mean_3d'], None) * (1 - use_template) * weight
            vertex_idx = smpl_x.lr_idx_to_hr_idx(torch.arange(smpl_x.vertex_num))
            loss['lap_scale'] = (self.lap_reg(assets['scale'], None).mean(1) + self.lap_reg_lr(assets['scale'][:,vertex_idx,:], None).mean(1)) * 1000000
            
            # rgb regularizers
            loss['lap_rgb'] = self.lap_reg(assets['rgb'], None) * 0.1
            loss['hand_rgb_tv_reg'] = self.tv_reg(renders['img_hand'], (torch.cat((renders['mask_hand'][:,0,None].repeat(1,3,1,1), renders['mask_hand'][:,1,None].repeat(1,3,1,1)),1) > 0.9)) * 100
            loss['hand_rgb_reg'] = (assets['rgb'][:,self.persona.is_hand,:] - torch.sigmoid(self.persona.rgb_aux)[None,self.persona.is_hand,:].detach()) ** 2
            loss['mouth_in_rgb_reg'] = (assets['rgb'][:,self.persona.is_mouth_in,:] - assets['rgb'][:,self.persona.is_mouth_out,:].mean(1)[:,None,:].detach()) ** 2
            return loss
        else:
            out = {}
            out['img'] = renders['img']
            out['img_refined'] = renders['img_refined']
            out['mask'] = (renders['mask'] > 0.9).float()
            out['mask_refined'] = (renders['mask_refined'] > 0.9).float()
            out['normal'] = renders['normal']
            out['normal_refined'] = renders['normal_refined']
            out['depth'] = renders['depth']
            out['depth_refined'] = renders['depth_refined']
            out['mean_3d'] = assets['mean_3d']
            out['mean_3d_refined'] = assets_refined['mean_3d']
            out['smplx_vert'] = smplx_verts['lr']
            return out

def get_model(smplx_params):
    persona = PERSONA()
    with torch.no_grad():
        persona.init()

    if smplx_params is not None:
        smplx_param_dict = SMPLXParamDict()
        with torch.no_grad():
            smplx_param_dict.init(smplx_params)
    else:
        smplx_param_dict = None

    model = Model(persona, smplx_param_dict)
    return model

