import torch
import argparse
from tqdm import tqdm
import numpy as np
from config import cfg
from base import Tester
import os
import os.path as osp
import random
import cv2
from glob import glob
from utils.smpl_x import smpl_x
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)

    # set dummy data, which will be replaced with data from checkpoint
    root_path = osp.join('..', 'data', 'subjects', cfg.subject_id)
    smpl_x.set_id_info(None, None, None)
    smpl_x.set_texture(None, None)
    tester._make_model()
    model = tester.model.module

    render_shape = (1024, 1024)
    zero_pose = torch.zeros((3)).float().cuda()
    smplx_param = {'root_pose': torch.FloatTensor([math.pi,0,0]).cuda(), \
                'body_pose': smpl_x.neutral_body_pose.view(-1).cuda(), \
                'jaw_pose': zero_pose, \
                'leye_pose': zero_pose, \
                'reye_pose': zero_pose, \
                'lhand_pose': torch.zeros((len(smpl_x.joint['part_idx']['lhand'])*3)).float().cuda(), \
                'rhand_pose': torch.zeros((len(smpl_x.joint['part_idx']['rhand'])*3)).float().cuda(), \
                'expr': torch.zeros((smpl_x.expr_param_dim)).float().cuda(), \
                'trans': torch.FloatTensor((0,3,3)).float().cuda()}
    cam_param = {'R': torch.FloatTensor(((1,0,0), (0,1,0), (0,0,1))).float().cuda(), \
                't': torch.zeros((3)).float().cuda(), \
                'focal': torch.FloatTensor((1500,1500)).cuda(), \
                'princpt': torch.FloatTensor((render_shape[1]/2, render_shape[0]/2)).cuda()}
    mesh_cam = torch.matmul(axis_angle_to_matrix(smplx_param['root_pose']), smpl_x.layer.v_template.cuda().permute(1,0)).permute(1,0) + smplx_param['trans'].view(1,3)
    at_point_cam = mesh_cam.mean(0)
    at_point = torch.matmul(torch.inverse(cam_param['R']), (at_point_cam - cam_param['t']))
    cam_pos = torch.matmul(torch.inverse(cam_param['R']), -cam_param['t'].view(3,1)).view(3)
    
    save_path = './neutral_pose'
    os.makedirs(save_path, exist_ok=True)
    view_num = 50
    for i in range(view_num):
        # make camera parameters
        azim = math.pi + math.pi*2*i/view_num # azim angle of the camera
        elev = -math.pi/12 
        dist = torch.sqrt(torch.sum((cam_pos - at_point)**2)) # distance between camera and mesh
        R, t = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=False, at=at_point[None,:], up=((0,1,0),)) 
        R = torch.inverse(R)
        cam_param_rot = {'R': R[0].cuda(), 't': t[0].cuda(), 'focal': cam_param['focal'], 'princpt': cam_param['princpt']}
        
        # render
        with torch.no_grad():
            asset, asset_refined, offset, vert_neutral_pose, vert = model.persona(smplx_param, cam_param)
            render = model.gaussian_renderer(asset, render_shape, cam_param_rot)['img']
            #render = model.gaussian_renderer(asset_refined, render_shape, cam_param_rot)['img']

        cv2.imwrite(osp.join(save_path, str(i) + '.png'), render.cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)

if __name__ == "__main__":
    main()
