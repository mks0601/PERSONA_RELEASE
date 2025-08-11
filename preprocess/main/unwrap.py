import argparse
import numpy as np
import cv2
from config import cfg
import torch
import torch.nn as nn
import json
import os
import os.path as osp
from utils.smpl_x import smpl_x
from utils.flame import flame
from glob import glob
from tqdm import tqdm
from base import Trainer
from pytorch3d.io import load_ply
from nets.layer import XY2UV

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--split', type=str, dest='split')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def main():
    args = parse_args()
    cfg.set_args(args.subject_id, split=args.split)
    
    root_path = osp.join('..', 'data', 'subjects', cfg.subject_id, cfg.split)
    smplx_xy2uv = XY2UV(smpl_x.vertex_uv, smpl_x.face_uv, cfg.smplx_uvmap_shape).cuda()
    smplx_texture_save, smplx_texture_mask_save = 0, 0
    with open(osp.join(root_path, 'segs', 'palette.json')) as f:
        seg_palette = np.array(json.load(f)['color'], dtype=np.float32)
        seg_num = len(seg_palette)
    smplx_seg_save, smplx_seg_mask_save = 0, 0
    img_path_list = glob(osp.join(root_path, 'images', '*.png'))
    for img_path in tqdm(img_path_list):
        frame_idx = int(img_path.split('/')[-1][:-4])

        ## smplx
        # load image, segmentation, and mask
        img = torch.FloatTensor(cv2.imread(img_path).astype(np.float32)[:,:,::-1].copy()).cuda().permute(2,0,1)/255
        seg_idx = torch.LongTensor(np.load(osp.join(root_path, 'segs', str(frame_idx) + '_seg.npy'))).cuda()
        seg_height, seg_width = seg_idx.shape
        seg = torch.zeros((seg_num,seg_height,seg_width)).float().cuda().scatter_(0,seg_idx[None],1) # index -> one hot
        mask_path = osp.join(root_path, 'masks', str(frame_idx) + '.png')
        mask = torch.FloatTensor(cv2.imread(mask_path).astype(np.float32)).cuda().permute(2,0,1)[0,None,:,:]/255

        # load camera parameter
        cam_param_path = osp.join(root_path, 'cam_params', str(frame_idx) + '.json')
        with open(cam_param_path) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
        
        # load mesh
        mesh_path = osp.join(root_path, 'smplx_optimized', 'meshes', str(frame_idx) + '.ply')
        vert, facee = load_ply(mesh_path)
        vert = vert.cuda()

        # unwrap to UV space
        img = torch.cat((img, mask))
        smplx_texture, smplx_texture_mask = smplx_xy2uv(img[None], vert[None], smpl_x.face, {k: v[None] for k,v in cam_param.items()})
        smplx_texture, smplx_texture_mask = smplx_texture[:,:3], (smplx_texture[:,3:]>0)*smplx_texture_mask
        smplx_texture *= smplx_texture_mask
        smplx_seg, smplx_seg_mask = smplx_xy2uv(seg[None], vert[None], smpl_x.face, {k: v[None] for k,v in cam_param.items()})
        smplx_seg *= smplx_texture_mask
        smplx_seg_mask *= smplx_texture_mask

        ## replace face from smplx with flame
        if cfg.split == 'captured':
            # load image (upsampled face patch)
            img_path = osp.join(root_path, 'face_patches', 'images', str(frame_idx) + '.png')
            img = torch.FloatTensor(cv2.imread(img_path).astype(np.float32)[:,:,::-1].copy()).cuda().permute(2,0,1)/255

            # modify camera parameter
            _, patch_height, patch_width = img.shape
            face_bbox_path = osp.join(root_path, 'face_patches', 'bbox', str(frame_idx) + '.json')
            with open(face_bbox_path) as f:
                face_bbox = np.array(json.load(f), dtype=np.float32) # xywh
            cam_param['focal'][0] = cam_param['focal'][0]/face_bbox[2]*patch_width
            cam_param['focal'][1] = cam_param['focal'][1]/face_bbox[3]*patch_height
            cam_param['princpt'][0] = (cam_param['princpt'][0]-face_bbox[0])/face_bbox[2]*patch_width
            cam_param['princpt'][1] = (cam_param['princpt'][1]-face_bbox[1])/face_bbox[3]*patch_height
            
            # replace face mesh
            mesh_path = osp.join(root_path, 'smplx_optimized', 'meshes', str(frame_idx) + '_flame.ply')
            vert_flame, facee = load_ply(mesh_path)
            vert_flame = vert_flame.cuda()
            vert[smpl_x.face_vertex_idx,:] = vert_flame

            # unwrap to UV space and replace face texture
            flame_texture, flame_texture_mask = smplx_xy2uv(img[None], vert[None], smpl_x.face, {k: v[None] for k,v in cam_param.items()})
            flame_texture_mask *= smpl_x.face_uv_mask[None,:,:,:].cuda()
            smplx_texture = smplx_texture*(1-flame_texture_mask) + flame_texture*flame_texture_mask
            smplx_texture_mask = ((smplx_texture_mask + flame_texture_mask) > 0).float()

        # save
        smplx_texture_save += smplx_texture[0].detach().cpu().numpy()
        smplx_texture_mask_save += smplx_texture_mask[0].detach().cpu().numpy()
        smplx_seg_save += smplx_seg[0].detach().cpu().numpy()
        smplx_seg_mask_save += smplx_seg_mask[0].detach().cpu().numpy()

    # save
    save_root_path = osp.join(root_path, 'smplx_optimized')
    os.makedirs(save_root_path, exist_ok=True)

    smplx_texture = (smplx_texture_save / (smplx_texture_mask_save + 1e-4) * 255).transpose(1,2,0)[:,:,::-1].astype(np.uint8)
    smplx_texture_mask = ((smplx_texture_mask_save > 0)*255).astype(np.uint8)[0]
    do_inpaint = (smplx_texture_mask == 0).astype(np.uint8)
    smplx_texture_inpaint = cv2.inpaint(smplx_texture, do_inpaint, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(osp.join(save_root_path, 'smplx_texture.png'), smplx_texture)
    cv2.imwrite(osp.join(save_root_path, 'smplx_texture_mask.png'), smplx_texture_mask)
    cv2.imwrite(osp.join(save_root_path, 'smplx_texture_inpaint.png'), smplx_texture_inpaint)

    smplx_seg = smplx_seg_save.argmax(0) # frequency -> index
    smplx_seg = seg_palette[smplx_seg]
    smplx_seg = smplx_seg[:,:,::-1].astype(np.uint8)
    smplx_seg_mask = ((smplx_seg_mask_save > 0)*255).astype(np.uint8)[0]
    do_inpaint = (smplx_seg_mask == 0).astype(np.uint8)
    smplx_seg_inpaint = cv2.inpaint(smplx_seg, do_inpaint, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(osp.join(save_root_path, 'smplx_seg.png'), smplx_seg)
    cv2.imwrite(osp.join(save_root_path, 'smplx_seg_mask.png'), smplx_seg_mask)
    cv2.imwrite(osp.join(save_root_path, 'smplx_seg_inpaint.png'), smplx_seg_inpaint)

if __name__ == "__main__":
    main()
