import torch
import os
import os.path as osp
from pytorch3d.io import save_ply
from glob import glob
from tqdm import tqdm
import smplx
import cv2
import json
import numpy as np
import argparse
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

def render_mesh(vert, face, cam_param, bkg, blend_ratio=1.0):
    vert = vert[None]
    face = face[None]
    cam_param = {k: v[None] for k,v in cam_param.items()}
    render_shape = (bkg.shape[0], bkg.shape[1]) # height, width

    batch_size, vertex_num = vert.shape[:2]
    textures = TexturesVertex(verts_features=torch.ones((batch_size,vertex_num,3)).float().cuda())
    vert = torch.stack((-vert[:,:,0], -vert[:,:,1], vert[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(vert, face, textures)

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
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--data_format', type=str, dest='data_format')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    assert args.data_format in ['image', 'video'], "Please set data_format."
    return args

args = parse_args()
root_path = args.root_path
data_format = args.data_format
smplx_layer = smplx.create('../third_modules/human_model_files', 'smplx', use_pca=False, flat_hand_mean=False, num_expression_coeffs=50).cuda()

smplx_param_path_list = glob(osp.join(root_path, 'smplx', 'params', '*.json'))
frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in smplx_param_path_list])
if data_format == 'video':
    img_height, img_width = cv2.imread(osp.join(root_path, 'images', str(frame_idx_list[0]) + '.png')).shape[:2]
    video_save = cv2.VideoWriter(osp.join(root_path, 'smplx.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))

for frame_idx in tqdm(frame_idx_list):
    # load smplx parameter
    with open(osp.join(root_path, 'smplx', 'params', str(frame_idx) + '.json')) as f:
        smplx_param = {k: torch.FloatTensor(v).view(1,-1).cuda() for k,v in json.load(f).items()}
    
    # get mesh
    with torch.no_grad():
        output = smplx_layer(global_orient=smplx_param['root_pose'], body_pose=smplx_param['body_pose'], jaw_pose=smplx_param['jaw_pose'], leye_pose=smplx_param['leye_pose'], reye_pose=smplx_param['reye_pose'], left_hand_pose=smplx_param['lhand_pose'], right_hand_pose=smplx_param['rhand_pose'], expression=smplx_param['expr'], betas=smplx_param['shape'], transl=smplx_param['trans'])
    vert = output.vertices[0].detach()

    # render mesh
    with open(osp.join(root_path, 'cam_params', str(frame_idx) + '.json')) as f:
        cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
    img = cv2.imread(osp.join(root_path, 'images', str(frame_idx) + '.png'))
    render = render_mesh(vert, torch.LongTensor(smplx_layer.faces).cuda(), cam_param, img)
    if data_format == 'video':
        frame = np.concatenate((img, render),1)
        frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
        video_save.write(frame.astype(np.uint8))
    elif args.data_format == 'image':
        save_path = osp.join(root_path, 'smplx', 'renders')
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(osp.join(save_path, str(frame_idx) + '.png'), render)

    # save mesh
    save_path = osp.join(root_path, 'smplx', 'meshes')
    os.makedirs(save_path, exist_ok=True)
    save_ply(osp.join(save_path, str(frame_idx) + '.ply'), vert.cpu(), torch.LongTensor(smplx_layer.faces))


