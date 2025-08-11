import argparse
import os
import os.path as osp
import torch
import numpy as np
import json
import cv2
from glob import glob
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from utils.vis import render_mesh
from utils.preprocessing import set_aspect_ratio, generate_patch_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    assert args.motion_path, 'Motion path for the animation is required.'
    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)

    # set dummy data, which will be replaced with data from checkpoint
    root_path = osp.join('..', 'data', 'subjects', cfg.subject_id)
    smpl_x.set_id_info(None, None, None)
    smpl_x.set_texture(None, None, None)
    tester._make_model()
    model = tester.model.module

    motion_path = args.motion_path
    if motion_path[-1] == '/':
        motion_name = motion_path[:-1].split('/')[-1]
    else:        
        motion_name = motion_path.split('/')[-1]
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', '*.json'))])
    render_shape = cv2.imread(osp.join(args.motion_path, 'images', str(frame_idx_list[0]) + '.png')).shape[:2]
   
    # load reference image
    img_ref = cv2.imread(osp.join(osp.join('..', 'data', 'subjects', cfg.subject_id, 'captured', 'images', '0.png')))
    bbox = set_aspect_ratio(np.array([0,0,img_ref.shape[1],img_ref.shape[0]], dtype=np.float32), extend_ratio=1.5, aspect_ratio=img_ref.shape[1]/img_ref.shape[0])
    img_ref, _, _ = generate_patch_image(img_ref, bbox, (render_shape[0], int(img_ref.shape[1]/img_ref.shape[0]*render_shape[0])))
        
    video_out = cv2.VideoWriter(cfg.subject_id + '_' + motion_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_ref.shape[1]+render_shape[1]*3, render_shape[0]))
    for frame_idx in tqdm(frame_idx_list):

        with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
        with open(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', str(frame_idx) + '.json')) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}
        
        # forward
        with torch.no_grad():
            asset, asset_refined, offset, vert_neutral_pose, vert = model.persona(smplx_param, cam_param)
            render = model.gaussian_renderer(asset, render_shape, cam_param)
            render = (render['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)
            render_refined = model.gaussian_renderer(asset_refined, render_shape, cam_param)
            render_refined = (render_refined['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)
            
        # smplx mesh render
        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(1,(len(smpl_x.joint['part_idx']['body'])-1)*3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(1,len(smpl_x.joint['part_idx']['lhand'])*3)
        rhand_pose = smplx_param['rhand_pose'].view(1,len(smpl_x.joint['part_idx']['rhand'])*3)
        expr = smplx_param['expr'].view(1,smpl_x.expr_param_dim)
        trans = smplx_param['trans'].view(1,3)
        shape = smpl_x.shape_param.cuda()[None]
        face_offset = smpl_x.face_offset.cuda()[None]
        joint_offset = smpl_x.joint_offset.cuda()[None]
        output = model.smplx_layer(global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, expression=expr, betas=shape, transl=trans, face_offset=face_offset, joint_offset=joint_offset)
        mesh = output.vertices[0]
        mesh_render = render_mesh(mesh, smpl_x.face, cam_param, np.ones((render_shape[0],render_shape[1],3), dtype=np.float32)*255).astype(np.uint8)
        
        # write texts
        font_size = 1.5
        thick = 3
        cv2.putText(img_ref, 'Reference image', (int(1/5*img_ref.shape[1]), int(0.05*img_ref.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
        cv2.putText(mesh_render, 'Target pose', (int(1/3*mesh_render.shape[1]), int(0.05*mesh_render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2)
        cv2.putText(render, 'Wo. Pose-driven deform.', (int(1/3*render.shape[1]), int(0.05*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
        cv2.putText(render_refined, 'PERSONA (Ours)', (int(1/3*render_refined.shape[1]), int(0.05*render_refined.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 
        out = np.concatenate((img_ref, mesh_render, render, render_refined),1).astype(np.uint8)
        
        # write frame
        out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.01), int(out.shape[0]*0.025)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, 2)
        video_out.write(out)
    
    video_out.release()
    
if __name__ == "__main__":
    main()
