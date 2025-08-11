import numpy as np
import cv2
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import argparse

import sys
cur_dir = osp.dirname(os.path.abspath(__file__))
root_dir = osp.join(cur_dir, '..', '..')
sys.path.insert(0, osp.join(root_dir, 'main'))
sys.path.insert(0, osp.join(root_dir, 'common'))
from utils.preprocessing import get_bbox, set_aspect_ratio, get_patch_img

def get_one_box(det_output):
    max_score = 0
    max_bbox = None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = score

    return max_bbox

def sanitize_bbox(bbox, img_height, img_width):
    xmin, ymin, width, height = bbox
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmin+width,img_width-1)
    ymax = min(ymin+height,img_height-1)
    bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin], dtype=np.float32)
    return bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--save_path', type=str, dest='save_path')
    parser.add_argument('--split', type=str, dest='split')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    assert args.save_path, "Please set save_path."
    assert args.split, "Please set split."
    return args

# get path
args = parse_args()
root_path = args.root_path
save_path = args.save_path
split = args.split
assert split in ['captured', 'test']
os.makedirs(save_path, exist_ok=True)
os.makedirs(osp.join(save_path, split), exist_ok=True)
os.makedirs(osp.join(save_path, split, 'images'), exist_ok=True)

# initialize human detector
det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
det_transform = T.Compose([T.ToTensor()])

# copy frame idx of the original dataset
if split == 'captured':
    with open(osp.join(save_path, split, 'frame_idx_orig.txt')) as f:
        lines = f.readlines()
    
    frame_idx_list = []
    for line in lines:
        data_split, capture_id, frame_idx = line.split()
        frame_idx_list.append({'data_split': data_split, 'capture_id': capture_id, 'frame_idx': int(frame_idx)})

elif split == 'test':
    f = open(osp.join(save_path, split, 'frame_idx_orig.txt'), 'w')
    frame_idx_list = []

    capture_path_list = glob(osp.join(root_path, split, 'Take*'))
    for capture_path in capture_path_list:
        capture_id = capture_path.split('/')[-1]
        _frame_idx_list = sorted([int(x.split('/')[-1][:-4].split('color_')[1]) for x in glob(osp.join(capture_path, 'render', 'image', '*.png'))])
        for frame_idx in _frame_idx_list:
            f.write(split + ' ' + capture_id + ' ' + str(frame_idx) + '\n')
            frame_idx_list.append({'data_split': split, 'capture_id': capture_id, 'frame_idx': frame_idx})

for i, data in enumerate(tqdm(frame_idx_list)):
    data_split, capture_id, frame_idx = data['data_split'], data['capture_id'], data['frame_idx']
    
    # load image
    img = cv2.imread(osp.join(root_path, data_split, capture_id, 'render', 'image', 'color_%06d.png' % frame_idx))
    img_height, img_width = img.shape[:2]

    # prepare bbox
    det_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    det_input = det_transform(img).cuda()
    det_output = det_model([det_input])[0]
    bbox = get_one_box(det_output) # xyxy
    bbox = np.array([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], dtype=np.float32) # xywh

    # crop human
    bbox = set_aspect_ratio(bbox, aspect_ratio=0.75, extend_ratio=1.25)
    bbox = sanitize_bbox(bbox, img_height, img_width)
    bbox = [int(x) for x in bbox]
    resize_shape = (bbox[3], bbox[2])
    img, _, _ = get_patch_img(img, bbox, resize_shape)

    # save cropped image, bbox for the crop, and the original image information
    cv2.imwrite(osp.join(save_path, split, 'images', str(i) + '.png'), img)
    with open(osp.join(save_path, split, 'images', str(i) + '_bbox_orig.json'), 'w') as f:
        json.dump(bbox, f)
    with open(osp.join(save_path, split, 'images', 'img_shape_orig.json'), 'w') as f:
        json.dump([img_height, img_width], f)

