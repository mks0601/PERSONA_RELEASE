import os
import os.path as osp
import sys

class Config:
    
    ## fitting
    kpt_proj_shape = (8, 8)
    depth_render_shape = (256, 256)
    smplx_uvmap_shape = (1024, 1024)
    face_patch_shape = (256, 256)
    lr_dec_factor = 10
    body_3d_size = 2 # meter
    batch_size = 256
    itr_opt_num = 250
    lr = 1e-2
    lr_dec_itr = [100]
    stage_itr = [100]
    end_epoch = 3

    ## others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')

    def set_args(self, subject_id, split):
        self.subject_id = subject_id
        self.split = split
        self.result_dir = osp.join(self.result_dir, subject_id)
        os.makedirs(self.result_dir, exist_ok=True)
 
    def set_stage(self, itr):
        if itr < self.stage_itr[0]:
            self.use_depthmap_loss = False
        else:
            self.use_depthmap_loss = True

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
