import torch
import argparse
from tqdm import tqdm
import numpy as np
from config import cfg
from base import Tester
import os
import os.path as osp
import cv2
from utils.smpl_x import smpl_x
from pytorch3d.io import save_ply
from torch.nn import functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--fit_pose_to_test', dest='fit_pose_to_test', action='store_true')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id, args.fit_pose_to_test)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    
    for itr, data in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(data, 'test')
        
        # save
        img = out['img'].cpu().numpy()
        img_refined = out['img_refined'].cpu().numpy()
        mask = out['mask'].cpu().numpy()
        mask_refined = out['mask_refined'].cpu().numpy()
        normal = out['normal'].cpu().numpy()
        normal_refined = out['normal_refined'].cpu().numpy()
        depth = out['depth'].cpu().numpy()
        depth_refined = out['depth_refined'].cpu().numpy()
        mean_3d = out['mean_3d'].cpu()
        mean_3d_refined = out['mean_3d_refined'].cpu()
        smplx_vert = out['smplx_vert'].cpu()
        batch_size = img.shape[0]
        for i in range(batch_size):
            split = data['split'][i]
            frame_idx = int(data['frame_idx'][i])
            save_root_path = cfg.result_dir
            os.makedirs(save_root_path, exist_ok=True)
            cv2.imwrite(osp.join(save_root_path, split + '_' + str(frame_idx) + '.png'), img[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, split + '_' + str(frame_idx) + '_refined.png'), img_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, split + '_' + str(frame_idx) + '_mask.png'), mask[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, split + '_' + str(frame_idx) + '_mask_refined.png'), mask_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, split + '_' + str(frame_idx) + '_normal.png'), normal[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, split + '_' + str(frame_idx) + '_normal_refined.png'), normal_refined[i].transpose(1,2,0)[:,:,::-1]*255)
            cv2.imwrite(osp.join(save_root_path, split + '_' + str(frame_idx) + '_gt.png'), data['img'][i].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255)
            save_ply(osp.join(save_root_path, split + '_' + str(frame_idx) + '_smplx.ply'), smplx_vert[i], torch.LongTensor(smpl_x.face.astype(np.int64)))
            save_ply(osp.join(save_root_path, split + '_' + str(frame_idx) + '.ply'), mean_3d[i], torch.LongTensor(smpl_x.face_upsampled.astype(np.int64)))
            save_ply(osp.join(save_root_path, split + '_' + str(frame_idx) + '_refined.ply'), mean_3d_refined[i], torch.LongTensor(smpl_x.face_upsampled.astype(np.int64)))
            np.save(osp.join(save_root_path, split + '_' + str(frame_idx) + '_normal.npy'), normal[i]*2-1)
            np.save(osp.join(save_root_path, split + '_' + str(frame_idx) + '_normal_refined.npy'), normal_refined[i]*2-1)
            np.save(osp.join(save_root_path, split + '_' + str(frame_idx) + '_depth.npy'), depth[i])
            np.save(osp.join(save_root_path, split + '_' + str(frame_idx) + '_depth_refined.npy'), depth_refined[i])


if __name__ == "__main__":
    main()
