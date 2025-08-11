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
from glob import glob
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
from base import Trainer
from utils.vis import render_mesh
from pytorch3d.io import save_ply

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def axis_angle_to_rotation_6d(x):
    return matrix_to_rotation_6d(axis_angle_to_matrix(x))

def rotation_6d_to_axis_angle(x):
    return matrix_to_axis_angle(rotation_6d_to_matrix(x))

def main():
    args = parse_args()
    cfg.set_args(args.subject_id)
    
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    # register initial smplx parameters
    smplx_params = {}
    for frame_idx in trainer.smplx_params.keys():
        smplx_params[frame_idx] = {}
        for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose']:
            smplx_params[frame_idx][key] = nn.Parameter(axis_angle_to_rotation_6d(trainer.smplx_params[frame_idx][key].cuda()))
        smplx_params[frame_idx]['expr'] = nn.Parameter(trainer.smplx_params[frame_idx]['expr'].cuda()) 
        smplx_params[frame_idx]['trans'] = nn.Parameter(trainer.smplx_params[frame_idx]['trans'].cuda()) 
    smplx_shape = nn.Parameter(torch.zeros((smpl_x.shape_param_dim)).float().cuda())

    for epoch in range(cfg.end_epoch):
        cfg.set_itr_opt_num(epoch)

        for itr_data, data in enumerate(trainer.batch_generator):
            batch_size = data['kpt_img'].shape[0]

            for itr_opt in range(cfg.itr_opt_num):
                cfg.set_stage(epoch, itr_opt)

                # optimizer
                if (epoch == 0) and (itr_opt == 0):
                    # smplx root pose and translatioin
                    optimizable_params = []
                    for frame_idx in data['frame_idx']:
                        for key in ['root_pose', 'trans']:
                            optimizable_params.append(smplx_params[int(frame_idx)][key])
                    trainer.get_optimizer(optimizable_params)
                elif ((epoch == 0) and (itr_opt == cfg.stage_itr[0])) or ((epoch > 0) and (itr_opt == 0)):
                    # all parameters
                    if epoch == (cfg.end_epoch - 1):
                        optimizable_params = [] # do not optimize shared parameters to make per-frame parameters consistent with the shared ones
                    else:
                        optimizable_params = [smplx_shape] # all shared parameters
                    for frame_idx in data['frame_idx']:
                        for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']: 
                            optimizable_params.append(smplx_params[int(frame_idx)][key])
                    trainer.get_optimizer(optimizable_params)

                # inputs
                smplx_inputs = {'shape': [smplx_shape for _ in range(batch_size)]}
                for frame_idx in data['frame_idx']:
                    for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                        if key not in smplx_inputs:
                            smplx_inputs[key] = [smplx_params[int(frame_idx)][key]]
                        else:
                            smplx_inputs[key].append(smplx_params[int(frame_idx)][key])
                for key in smplx_inputs.keys():
                    smplx_inputs[key] = torch.stack(smplx_inputs[key])

                # forwrad
                trainer.set_lr(itr_opt)
                trainer.optimizer.zero_grad()
                loss, out = trainer.model(smplx_inputs, data, return_output=(itr_opt==cfg.itr_opt_num-1))
                loss = {k:loss[k].mean() for k in loss}
                
                # backward
                sum(loss[k] for k in loss).backward()
                trainer.optimizer.step()
                print(cfg.result_dir)

                # log
                screen = [
                    'epoch %d/%d itr_data %d/%d itr_opt %d/%d:' % (epoch, cfg.end_epoch, itr_data, trainer.itr_per_epoch, itr_opt, cfg.itr_opt_num),
                    'lr: %g' % (trainer.get_lr()),
                    ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                trainer.logger.info(' '.join(screen))
            
            # save
            if epoch != (cfg.end_epoch-1):
                continue

            save_root_path = osp.join(cfg.result_dir, 'smplx_optimized')
            os.makedirs(save_root_path, exist_ok=True)
            smplx_mesh_cam = out['smplx_mesh_cam'].detach().cpu()
            smplx_trans = out['smplx_trans'].detach().cpu().numpy()
            for i in range(batch_size):
                frame_idx = int(data['frame_idx'][i])

                # mesh
                save_path = osp.join(save_root_path, 'meshes')
                os.makedirs(save_path, exist_ok=True)
                save_ply(osp.join(save_path, str(frame_idx) + '.ply'), torch.FloatTensor(smplx_mesh_cam[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())

                # smplx parameter
                save_path = osp.join(save_root_path, 'smplx_params')
                os.makedirs(save_path, exist_ok=True)
                with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
                    json.dump({'root_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['root_pose'].detach().cpu()).numpy().tolist(), \
                            'body_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['body_pose'].detach().cpu()).numpy().tolist(), \
                            'jaw_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['jaw_pose'].detach().cpu()).numpy().tolist(), \
                            'leye_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['leye_pose'].detach().cpu()).numpy().tolist(), \
                            'reye_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['reye_pose'].detach().cpu()).numpy().tolist(), \
                            'lhand_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['lhand_pose'].detach().cpu()).numpy().tolist(), \
                            'rhand_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['rhand_pose'].detach().cpu()).numpy().tolist(), \
                            'expr': smplx_params[frame_idx]['expr'].detach().cpu().numpy().tolist(), \
                            'trans': smplx_trans[i].tolist()}, f)
                if (itr_data == 0) and (i == 0):
                    # shape parameter
                    with open(osp.join(save_root_path, 'shape_param.json'), 'w') as f:
                        json.dump(smplx_shape.detach().cpu().numpy().tolist(), f)

if __name__ == "__main__":
    main()
