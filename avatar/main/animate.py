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
from utils.preprocessing import set_aspect_ratio, get_patch_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    parser.add_argument('--use_bkg', dest='use_bkg', action='store_true')
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
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx', 'params', '*.json'))])
    render_shape = cv2.imread(osp.join(args.motion_path, 'images', str(frame_idx_list[0]) + '.png')).shape[:2]
   
    # load reference image
    img_ref = cv2.imread(osp.join(osp.join('..', 'data', 'subjects', cfg.subject_id, 'captured', 'images', '0.png')))
    bbox = set_aspect_ratio(np.array([0,0,img_ref.shape[1],img_ref.shape[0]], dtype=np.float32), extend_ratio=1.5, aspect_ratio=img_ref.shape[1]/img_ref.shape[0])
    img_ref, _, _ = get_patch_image(img_ref, bbox, (render_shape[0], int(img_ref.shape[1]/img_ref.shape[0]*render_shape[0])), bordervalue=(255,255,255))
        
    video_out = cv2.VideoWriter(cfg.subject_id + '_' + motion_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_ref.shape[1]+render_shape[1], render_shape[0]))
    for frame_idx in tqdm(frame_idx_list):

        with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
        with open(osp.join(args.motion_path, 'smplx', 'params', str(frame_idx) + '.json')) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}

        # forward
        with torch.no_grad():
            asset, asset_refined, offset, vert_neutral_pose, vert = model.persona(smplx_param, cam_param)
            render_refined = model.gaussian_renderer(asset_refined, render_shape, cam_param)
            render_refined = (render_refined['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)
            if args.use_bkg:
                bkg = cv2.imread(osp.join(motion_path, 'images_bkg', str(frame_idx) + '.png'))
                asset_refined_mask = {k: torch.ones_like(v) if k == 'rgb' else v for k,v in asset_refined.items()}
                mask_refined = model.gaussian_renderer(asset_refined_mask, render_shape, cam_param, bg=torch.zeros((3)).float().cuda())
                mask_refined = mask_refined['img'].cpu().numpy().transpose(1,2,0)
                render_refined = render_refined*mask_refined + bkg*(1-mask_refined)
        
        # write frame
        out = np.concatenate((img_ref, render_refined),1).astype(np.uint8)
        out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.025), int(out.shape[0]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, 2)
        video_out.write(out)
    
    video_out.release()
    
if __name__ == "__main__":
    main()
