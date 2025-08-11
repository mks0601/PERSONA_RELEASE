import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from config import cfg
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PointLights,
DirectionalLights,
PerspectiveCameras,
Materials,
SoftPhongShader,
RasterizationSettings,
MeshRenderer,
MeshRendererWithFragments,
MeshRasterizer,
TexturesVertex)
import torch
from plyfile import PlyData, PlyElement
from utils.transforms import RGB2SH

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def render_mesh(mesh, face, cam_param, bkg, blend_ratio=1.0):
    mesh = mesh.cuda()[None,:,:]
    face = torch.LongTensor(face.astype(np.int64)).cuda()[None,:,:]
    cam_param = {k: v.cuda()[None,:] for k,v in cam_param.items()}
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
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
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

def save_ply(save_path, xyz, scale, rotation, opacity, shs, use_rgb):
    
    def construct_list_of_attributes(shs, scale, rotation):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = shs[:, :1]
        features_rest = shs[:, 1:]

        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(scale.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    if use_rgb:
        shs = RGB2SH(shs)
        shs = shs[:,None,:].repeat(1,2,1)
        shs[:,1:,:] = 0
    else:
        shs = rgb

    features_dc = shs[:, :1]
    features_rest = shs[:, 1:]

    f_dc = (
        features_dc.float().detach().flatten(start_dim=1).contiguous().cpu().numpy()
    )
    f_rest = (
        features_rest.float()
        .detach()
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = (
        torch.logit(opacity, eps=1e-4)
        .detach()
        .cpu()
        .numpy()
    )

    scale = np.log(scale.detach().cpu().numpy())
    rotation = rotation.detach().cpu().numpy()

    dtype_full = [
        (attribute, "f4") for attribute in construct_list_of_attributes(shs, scale, rotation)
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(save_path)

