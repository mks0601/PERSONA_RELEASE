# use original images (without crop and resize) following https://github.com/aipixel/GaussianAvatar/blob/main/eval.py

import cv2
import torch
import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, dest='output_path')
    args = parser.parse_args()
    assert args.output_path, "Please set output_path."
    return args

# get path
args = parse_args()
output_path = args.output_path
if output_path[-1] == '/':
    output_path = output_path[:-1]
subject_id = output_path.split('/')[-1].split('_fit_pose_to_test')[0] # ../output/result/bike_fit_pose_to_test -> bike
data_root_path = osp.join('..', 'data', 'subjects', subject_id, 'test')

results = {'psnr': [], 'ssim': [], 'lpips': []}
psnr = PeakSignalNoiseRatio(data_range=1).cuda()
ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()

with open(osp.join(data_root_path, 'images', 'img_shape_orig.json')) as f:
    img_height_orig, img_width_orig = json.load(f)

frame_idx_list = [int(x.split('/')[-1][:-4]) for x in glob(osp.join(data_root_path, 'images', '*.png'))]
for frame_idx in tqdm(frame_idx_list):
    
    # bbox
    with open(osp.join(data_root_path, 'images', str(frame_idx) + '_bbox_orig.json')) as f:
        bbox = json.load(f)

    # output image
    out_path = osp.join(output_path, 'test_' + str(frame_idx) + '_refined.png')
    out = cv2.imread(out_path)[:,:,::-1]/255.
    _out = np.ones((img_height_orig, img_width_orig, 3), dtype=np.float32)
    _out[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :] = out
    out = torch.FloatTensor(_out).permute(2,0,1)[None,:,:,:].cuda()
    
    # gt image
    gt_path = osp.join(data_root_path, 'images', str(frame_idx) + '.png')
    gt = cv2.imread(gt_path)[:,:,::-1]/255.
    _gt = np.ones((img_height_orig, img_width_orig, 3), dtype=np.float32)
    _gt[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :] = gt
    gt = torch.FloatTensor(_gt).permute(2,0,1)[None,:,:,:].cuda()
    
    # gt mask
    mask_path = osp.join(data_root_path, 'masks', str(frame_idx) + '.png')
    mask = cv2.imread(mask_path)/255
    _mask = np.ones((img_height_orig, img_width_orig, 3), dtype=np.float32)
    _mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :] = mask
    mask = torch.FloatTensor(_mask).permute(2,0,1)[None,:,:,:].cuda()
    
    # exclude background pixels
    out = out * mask + 1 * (1 - mask)
    gt = gt * mask + 1 * (1 - mask)

    results['psnr'].append(psnr(out, gt))
    results['ssim'].append(ssim(out, gt))
    results['lpips'].append(lpips(out*2-1, gt*2-1)) # normalize to [-1,1]

print('output path: ' + output_path)
print('subject_id: ' + subject_id)
print({k: torch.FloatTensor(v).mean() for k,v in results.items()})

