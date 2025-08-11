import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.inference_utils import non_max_suppression
from glob import glob
import json
from pytorch3d.io import save_ply

os.environ["PYOPENGL_PLATFORM"] = "egl"
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRenderer,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)

def render_mesh(mesh, face, cam_param, bkg, blend_ratio=1.0):
    mesh = torch.FloatTensor(mesh).cuda()[None,:,:]
    face = torch.LongTensor(face.astype(np.int64)).cuda()[None,:,:]
    cam_param = {k: torch.FloatTensor(v).cuda()[None,:] for k,v in cam_param.items()}
    render_shape = (bkg.shape[0], bkg.shape[1]) # height, width

    batch_size, vertex_num = mesh.shape[:2]
    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face, textures)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    lights = PointLights(device='cuda')
    shader = SoftPhongShader(device='cuda', cameras=cameras, lights=lights)
    materials = Materials(
	device='cuda',
	specular_color=[[0.0, 0.0, 0.0]],
	shininess=0.0
    )

    # render
    with torch.no_grad():
        renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
        images, fragments = renderer(mesh, materials=materials)
 
    # background masking
    is_bkg = (fragments.zbuf <= 0).float().cpu().numpy()[0]
    render = images[0,:,:,:3].cpu().numpy()
    fg = render * blend_ratio + bkg/255 * (1 - blend_ratio)
    render = fg * (1 - is_bkg) * 255 + bkg * is_bkg
    return render   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--data_format', type=str)
    args = parser.parse_args()
    assert args.root_path
    assert args.data_format in ['image', 'video']
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cur_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', 'smplest_x_h', 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', 'smplest_x_h', 'smplest_x_h.pth.tar')

    root_path = args.root_path
    if root_path[-1] == '/':
        root_path = root_path[:-1]
    subject_id = root_path.split('/')[-1]
    exp_name = f'inference_{subject_id}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(cur_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    img_path_list = glob(osp.join(root_path, 'images', '*'))
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
    img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]
    if args.data_format == 'video':
        video_save = cv2.VideoWriter(osp.join(root_path, 'smplx_optimized.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
    for frame_idx in tqdm(frame_idx_list):
        
        # prepare input image
        img_path = osp.join(root_path, 'images', str(frame_idx) + '.png')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=00, 
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()
        if len(yolo_bbox)<1:
            continue
        yolo_bbox = yolo_bbox[0]
        yolo_bbox_xywh = np.zeros((4))
        yolo_bbox_xywh[0] = yolo_bbox[0]
        yolo_bbox_xywh[1] = yolo_bbox[1]
        yolo_bbox_xywh[2] = abs(yolo_bbox[2] - yolo_bbox[0])
        yolo_bbox_xywh[3] = abs(yolo_bbox[3] - yolo_bbox[1])
        
        # xywh
        bbox = process_bbox(bbox=yolo_bbox_xywh, 
                            img_width=original_img_width, 
                            img_height=original_img_height, 
                            input_img_shape=cfg.model.input_img_shape, 
                            ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
        img, _, _ = generate_patch_image(cvimg=original_img, 
                                            bbox=bbox, 
                                            scale=1.0, 
                                            rot=0.0, 
                                            do_flip=False, 
                                            out_shape=cfg.model.input_img_shape)
            
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')
        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

        # render mesh
        focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                 cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
        princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                   cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
        vis_img = render_mesh(mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, original_img[:,:,::-1])

        # save camera parameter
        os.makedirs(osp.join(root_path, 'cam_params'), exist_ok=True)
        with open(osp.join(root_path, 'cam_params', str(frame_idx) + '.json'), 'w') as f:
            json.dump({'R': torch.eye(3).float().tolist(), 't': torch.zeros((3)).float().tolist(), 'focal': focal, 'princpt': princpt}, f)

        # save rendered image
        if args.data_format == 'video':
            frame = np.concatenate((original_img[:,:,::-1], vis_img),1)
            frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
            video_save.write(frame.astype(np.uint8))
        elif args.data_format == 'image':
            save_path = osp.join(root_path, 'smplx_optimized', 'renders')
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '.png'), vis_img)

        # save mesh
        save_path = osp.join(root_path, 'smplx_optimized', 'meshes')
        os.makedirs(save_path, exist_ok=True)
        save_ply(osp.join(save_path, str(frame_idx) + '.ply'), torch.FloatTensor(mesh), torch.LongTensor(smpl_x.face))

        # save smplx parameters
        save_path = osp.join(root_path, 'smplx_optimized', 'smplx_params')
        os.makedirs(save_path, exist_ok=True)
        root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
        body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0] 
        lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0] 
        rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0] 
        jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0] 
        shape = out['smplx_shape'].detach().cpu().numpy()[0]
        expr = out['smplx_expr'].detach().cpu().numpy()[0] 
        trans = out['cam_trans'].detach().cpu().numpy()[0]
        with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
            json.dump({'root_pose': root_pose.reshape(-1).tolist(), \
                    'body_pose': body_pose.reshape(-1,3).tolist(), \
                    'lhand_pose': lhand_pose.reshape(-1,3).tolist(), \
                    'rhand_pose': rhand_pose.reshape(-1,3).tolist(), \
                    'leye_pose': [0,0,0],\
                    'reye_pose': [0,0,0],\
                    'jaw_pose': jaw_pose.reshape(-1).tolist(), \
                    'shape': shape.reshape(-1).tolist(), \
                    'expr': expr.reshape(-1).tolist(),
                    'trans': trans.reshape(-1).tolist()}, f)

    # for the animation compatibility
    os.system('ln -s ' + osp.join(root_path, 'smplx_optimized', 'meshes') + ' ' + osp.join(root_path, 'smplx_optimized', 'meshes_smoothed'))
    os.system('ln -s ' + osp.join(root_path, 'smplx_optimized', 'renders') + ' ' + osp.join(root_path, 'smplx_optimized', 'renders_smoothed'))
    os.system('ln -s ' + osp.join(root_path, 'smplx_optimized', 'smplx_params') + ' ' + osp.join(root_path, 'smplx_optimized', 'smplx_params_smoothed'))

if __name__ == "__main__":
    main()
