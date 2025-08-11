import torch
import numpy as np
import math
from torch.nn import functional as F

def change_kpt_name(src_kpt, src_name, dst_name):
    src_kpt_num = len(src_name)
    dst_kpt_num = len(dst_name)

    new_kpt = np.zeros(((dst_kpt_num,) + src_kpt.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_kpt[dst_idx] = src_kpt[src_idx]

    return new_kpt

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def get_neighbor(vertex_num, face, neighbor_max_num=10):
    adj = {i: set() for i in range(vertex_num)}
    for i in range(len(face)):
        for idx in face[i]:
            adj[idx] |= set(face[i]) - set([idx])

    neighbor_idxs = np.tile(np.arange(vertex_num)[:,None], (1, neighbor_max_num))
    neighbor_weights = np.zeros((vertex_num, neighbor_max_num), dtype=np.float32)
    for idx in range(vertex_num):
        neighbor_num = min(len(adj[idx]), neighbor_max_num)
        neighbor_idxs[idx,:neighbor_num] = np.array(list(adj[idx]))[:neighbor_num]
        neighbor_weights[idx,:neighbor_num] = -1.0 / neighbor_num

    neighbor_idxs, neighbor_weights = torch.from_numpy(neighbor_idxs), torch.from_numpy(neighbor_weights)
    return neighbor_idxs, neighbor_weights

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z), 1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z), 1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def get_view_matrix(R, t):
    Rt = torch.cat((R, t.view(3,1)),1)
    view_matrix = torch.cat((Rt, torch.FloatTensor([0,0,0,1]).cuda().view(1,4)))
    return view_matrix

def get_proj_matrix(focal, princpt, img_shape, z_near, z_far, z_sign):
    fov = get_fov(focal, img_shape)
    fovY = fov[1]
    fovX = fov[0]
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right
    
    # take princpt into account (it could be not the image center)
    offset_x = princpt[0] - (img_shape[1]/2)
    offset_x = (offset_x/focal[0])*z_near
    offset_y = princpt[1] - (img_shape[0]/2)
    offset_y = (offset_y/focal[1])*z_near

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    proj_matrix = torch.zeros(4, 4).float().cuda()
    proj_matrix[0, 0] = 2.0 * z_near / (right - left)
    proj_matrix[1, 1] = 2.0 * z_near / (top - bottom)
    proj_matrix[0, 2] = (right + left) / (right - left)
    proj_matrix[1, 2] = (top + bottom) / (top - bottom)
    proj_matrix[3, 2] = z_sign
    #proj_matrix[2, 2] = z_sign * z_far / (z_far - z_near) # original 3DGS. has a minor bug
    #proj_matrix[2, 3] = -(z_far * z_near) / (z_far - z_near) # original 3DGS. has a minor bug
    proj_matrix[2, 2] = z_sign * (z_near + z_far) / (z_far - z_near)
    proj_matrix[2, 3] = -(2 * z_far * z_near) / (z_far - z_near)
    return proj_matrix

def get_fov(focal, img_shape):
    fov_x = 2 * torch.atan(img_shape[1] / (2 * focal[0]))
    fov_y = 2 * torch.atan(img_shape[0] / (2 * focal[1]))
    fov = torch.FloatTensor([fov_x, fov_y]).cuda()
    return fov

def get_covariance_matrix(scale, rotation):
    point_num = scale.shape[0]
    S = torch.zeros((point_num, 3, 3)).float().cuda()
    S[:,0,0] = scale[:,0]
    S[:,1,1] = scale[:,1]
    S[:,2,2] = scale[:,2]
    RS = torch.bmm(rotation, S)
    covariance = torch.bmm(RS, RS.permute(0,2,1))
    return covariance

