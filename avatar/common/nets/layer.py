import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, OrthographicCameras, RasterizationSettings, MeshRasterizer, TexturesVertex

def make_linear_layers(feat_dims, relu_final=True, use_gn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_gn:
                layers.append(nn.GroupNorm(4, feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def rasterize(vert, face, cam_param, render_shape, bin_size=None):
    batch_size = vert.shape[0]
    face = torch.from_numpy(face).cuda()[None,:,:].repeat(batch_size,1,1)
    vert = torch.stack((-vert[:,:,0], -vert[:,:,1], vert[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(vert, face)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                principal_point=cam_param['princpt'],
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=bin_size)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    fragments = rasterizer(mesh)
    return fragments

def rasterize_to_uv(vertex_uv, face_uv, uvmap_shape):
    # scale UV coordinates to uvmap_shape
    vertex_uv = torch.stack((vertex_uv[:,:,0] * uvmap_shape[1], vertex_uv[:,:,1] * uvmap_shape[0]),2)
    vertex_uv = torch.cat((vertex_uv, torch.ones_like(vertex_uv[:,:,:1])),2) # add dummy depth
    vertex_uv = torch.stack((-vertex_uv[:,:,0], -vertex_uv[:,:,1], vertex_uv[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(vertex_uv, face_uv)

    cameras = OrthographicCameras(
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(uvmap_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=uvmap_shape, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    outputs = rasterizer(mesh)
    return outputs

class MeshRenderer(nn.Module):
    def __init__(self):
        super(MeshRenderer, self).__init__()

    def forward(self, vert_mask, vert, face, cam_param, render_shape):
        batch_size = vert.shape[0]
        render_height, render_width = render_shape

        # rasterize
        vert = torch.bmm(cam_param['R'], vert.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3) # world coordinate -> camera coordinate
        fragments = rasterize(vert, face, cam_param, render_shape)

        # render texture
        vert_rgb = vert_mask[:,:,None].float()
        face = torch.LongTensor(face).cuda()
        render = TexturesVertex(vert_rgb).sample_textures(fragments, face)[0,:,:,0,:].permute(2,0,1) # 1, render_shape[0], render_shape[1]

        # fg mask
        pix_to_face = fragments.pix_to_face # batch_size, render_height, render_width, faces_per_pixel. invalid: -1
        pix_to_face_xy = pix_to_face[:,:,:,0] # Note: this is a packed representation
        is_fg = (pix_to_face_xy != -1).float()
        is_fg = is_fg[:,None,:,:]
        render = render * is_fg
        return render
