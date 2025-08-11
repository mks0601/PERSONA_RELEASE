import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from tqdm import tqdm
import numpy as np
import smplx
import os.path as osp

def create_diffused_skinning_field(
    verts: torch.Tensor,
    smplx_weights: torch.Tensor,
    grid_res: int = 64,
    smooth_iters: int = 100,
    proximity_thresh: float = 0.02,
    expansion_factor: float = 0.3
):
    joint_num = smplx_weights.shape[1]
    finger_joint_idx = list(range(25, 55))  # SMPL-X 손가락 joints
    non_finger_idx = [i for i in range(joint_num) if i not in finger_joint_idx]

    # extended bounding box
    vmin, vmax = verts.min(0).values, verts.max(0).values
    margin = expansion_factor * (vmax - vmin)
    vmin = vmin - margin
    vmax = vmax + margin

    # create 3D voxel grid
    coords = [torch.linspace(vmin[i], vmax[i], grid_res).cuda() for i in range(3)]
    xx, yy, zz = torch.meshgrid(*coords, indexing='ij')
    grid_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [D^3, 3]

    # find points close to surface 
    dists = knn_points(grid_points[None], verts[None], K=1).dists[0, :, 0]
    surface_mask = dists < (proximity_thresh ** 2)
    surface_points = grid_points[surface_mask]

    # get weights from nn
    nn_idx = knn_points(surface_points[None], verts[None], K=1).idx[0, :, 0]
    surface_weights = smplx_weights[nn_idx]  # [Ns, J]

    # initialize weight field
    field = torch.zeros((joint_num, grid_res, grid_res, grid_res)).cuda()
    mask = surface_mask.reshape(grid_res, grid_res, grid_res)
    field[:, mask] = surface_weights.T

    # exclude finger
    kernel = torch.tensor([[[0,0,0],[0,1,0],[0,0,0]],
                           [[0,1,0],[1,-6,1],[0,1,0]],
                           [[0,0,0],[0,1,0],[0,0,0]]],
                          dtype=torch.float32).cuda()[None, None].repeat(len(non_finger_idx), 1, 1, 1, 1)

    field_diffuse = field[non_finger_idx].clone()
    fixed_val = torch.zeros_like(field_diffuse)
    fixed_val[:, mask] = surface_weights[:, non_finger_idx].T
    fixed_mask = mask[None].repeat(len(non_finger_idx), 1, 1, 1)

    for _ in tqdm(range(smooth_iters), desc="Laplacian smoothing"):
        field_diffuse_pad = F.pad(field_diffuse, (1,1,1,1,1,1), mode='replicate')
        lap = F.conv3d(field_diffuse_pad[None], kernel, groups=len(non_finger_idx))[0]
        field_diffuse = field_diffuse + 0.1 * lap
        field_diffuse = torch.where(fixed_mask, fixed_val, field_diffuse)

    field[non_finger_idx] = field_diffuse

    # normalize
    field = torch.clamp(field, min=0)
    field = field / (field.sum(0, keepdim=True) + 1e-8)
    field = field.permute(1, 2, 3, 0).contiguous()  # [D, D, D, J]

    return field, coords

# Main
if __name__ == "__main__":
    smplx_layer = smplx.create(osp.join('..', '..', 'common/utils/human_model_files'), 'smplx')

    # joint set
    joint = {
    'num': 55, # 22 (body joints) + 3 (face joints) + 30 (hand joints)
    'name':
    ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
    'Jaw', 'L_Eye', 'R_Eye', # face joints
    'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
    'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3' # right hand joints
    )
    }
    joint['root_idx'] = joint['name'].index('Pelvis')
    joint['part_idx'] = \
    {'body': range(joint['name'].index('Pelvis'), joint['name'].index('R_Wrist')+1),
    'face': range(joint['name'].index('Jaw'), joint['name'].index('R_Eye')+1),
    'lhand': range(joint['name'].index('L_Index_1'), joint['name'].index('L_Thumb_3')+1),
    'rhand': range(joint['name'].index('R_Index_1'), joint['name'].index('R_Thumb_3')+1)}
    neutral_body_pose = torch.zeros((len(joint['part_idx']['body'])-1,3)) # 大 pose in axis-angle representation (body pose without root joint)
    neutral_body_pose[0] = torch.FloatTensor([0, 0, 1/3])
    neutral_body_pose[1] = torch.FloatTensor([0, 0, -1/3])
    
    verts = smplx_layer(body_pose=neutral_body_pose.view(1,-1)).vertices[0].detach().cuda()
    smplx_weights = smplx_layer.lbs_weights.cuda()

    skinning_field, grid_coords = create_diffused_skinning_field(
        verts=verts,
        smplx_weights=smplx_weights,
        grid_res=128,
        smooth_iters=100,
        proximity_thresh=0.02,
        expansion_factor=0.3
    )

    np.save("diffused_skinning_weights.npy", skinning_field.cpu().numpy())
    np.save("skinning_grid_coords_x.npy", grid_coords[0].cpu().numpy())
    np.save("skinning_grid_coords_y.npy", grid_coords[1].cpu().numpy())
    np.save("skinning_grid_coords_z.npy", grid_coords[2].cpu().numpy())
