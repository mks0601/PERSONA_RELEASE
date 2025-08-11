import os
import os.path as osp
import sys

class Config:
    
    ## shape
    smplx_uvmap_shape = (1024, 1024) # height, width
    triplane_shape_3d = (2, 2, 2) # meter
    triplane_shape = (32, 128, 128) # feat_dim, height, width
    face_patch_shape = (256, 256) # height, width

    ## train
    lr = 1e-3 
    smplx_param_lr = 1e-3
    boundary_thr = 0.02
    end_epoch = 5
    img_kernel_ratio = 0.015 
    patch_kernel_ratio = 0.06

    ## loss functions
    rgb_loss_weight = 0.8
    ssim_loss_weight = 0.2
    lpips_loss_weight = 1.0
    depth_loss_weight = 0.01

    ## others
    num_thread = 8
    num_gpus = 1
    batch_size = 1 # Gaussian splatting renderer only supports batch_size==1

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join('..', 'common', 'utils', 'human_model_files')
    sw_path = osp.join('..', 'tools', 'diffused_skinning_weights')

    def set_args(self, subject_id, fit_pose_to_test=False):
        self.subject_id = subject_id
        self.fit_pose_to_test = fit_pose_to_test
        if self.fit_pose_to_test:
            self.model_dir = osp.join(self.model_dir, subject_id + '_fit_pose_to_test')
            self.result_dir = osp.join(self.result_dir, subject_id + '_fit_pose_to_test')
        else:
            self.model_dir = osp.join(self.model_dir, subject_id)
            self.result_dir = osp.join(self.result_dir, subject_id)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
