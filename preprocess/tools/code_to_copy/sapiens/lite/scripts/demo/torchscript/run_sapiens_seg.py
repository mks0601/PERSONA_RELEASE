import os
import os.path as osp
from glob import glob
import numpy as np
import cv2
import json
import argparse

def make_video(video_output_path, output_path):
    frame_idx_list = sorted([int(x.split('/')[-1].split('_')[0]) for x in glob(osp.join(output_path, '*_vis.png'))])
    img_height, img_width = cv2.imread(osp.join(output_path, str(frame_idx_list[0]) + '_vis.png')).shape[:2]
    video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width, img_height))
    for frame_idx in frame_idx_list:
        frame = cv2.imread(osp.join(output_path, str(frame_idx) + '_vis.png'))
        frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
        video.write(frame.astype(np.uint8))
    video.release()

def remove_orig(output_path):
    img_path_list = [x for x in glob(osp.join(output_path, '*.png')) if 'vis.png' not in x]
    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img_width = img.shape[1]
        img = img[:,img_width//2:,:]
        cv2.imwrite(img_path, img)

NAME = [
'Background',
'Apparel',
'Face_Neck',
'Hair',
'Left_Foot',
'Left_Hand',
'Left_Lower_Arm',
'Left_Lower_Leg',
'Left_Shoe',
'Left_Sock',
'Left_Upper_Arm',
'Left_Upper_Leg',
'Lower_Clothing',
'Right_Foot',
'Right_Hand',
'Right_Lower_Arm',
'Right_Lower_Leg',
'Right_Shoe',
'Right_Sock',
'Right_Upper_Arm',
'Right_Upper_Leg',
'Torso',
'Upper_Clothing',
'Lower_Lip',
'Upper_Lip',
'Lower_Teeth',
'Upper_Teeth',
'Tongue'
]

COLOR = [
    [50, 50, 50],
    [255, 218, 0],
    [128, 200, 255],
    [255, 0, 109],
    [189, 0, 204],
    [255, 0, 218],
    [0, 160, 204],
    [0, 255, 145],
    [204, 0, 131],
    [182, 0, 255],
    [255, 109, 0],
    [0, 255, 255],
    [72, 0, 255],
    [204, 131, 0],
    [255, 0, 0],
    [72, 255, 0],
    [189, 204, 0],
    [182, 255, 0],
    [102, 0, 204],
    [32, 72, 204],
    [0, 145, 255],
    [14, 204, 0],
    [0, 128, 72],
    [235, 205, 119],
    [115, 227, 112],
    [157, 113, 143],
    [132, 93, 50],
    [82, 21, 114],
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--vis_format', type=str, dest='vis_format')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    assert args.vis_format, "Please set vis_format."
    return args

args = parse_args()
root_path = args.root_path
vis_format = args.vis_format
input_path = './input'
output_path = './output_seg'
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# prepare input images
os.system('rm -rf ' + osp.join(input_path, '*'))
os.system('cp ' + osp.join(root_path, 'images', '*') + ' ' + osp.join(input_path, '.'))
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(input_path, '*.png'))])

# segmentation
os.system('rm -rf ' + osp.join(output_path, '*'))
os.system('./seg.sh')
os.system('rm ' + osp.join(output_path, '*vis*.npy'))
remove_orig(output_path)
os.makedirs(osp.join(root_path, 'segs'), exist_ok=True)
if vis_format == 'image':
    os.system('mv ' + osp.join(output_path, '*') + ' ' + osp.join(root_path, 'segs', '.'))
else:
    make_video(osp.join(root_path, 'segs.mp4'), output_path)
    os.system('rm ' + osp.join(output_path, '*_vis.png'))
    os.system('mv ' + osp.join(output_path, '*') + ' ' + osp.join(root_path, 'segs', '.'))

with open(osp.join(root_path, 'segs', 'palette.json'), 'w') as f:
    json.dump({'name': NAME, 'color': COLOR}, f)
