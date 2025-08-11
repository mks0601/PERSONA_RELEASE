from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import argparse

def get_bbox(kpt_img, kpt_valid, extend_ratio=1.2):
    x_img, y_img = kpt_img[:,0], kpt_img[:,1]
    x_img = x_img[kpt_valid==1]; y_img = y_img[kpt_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--vis_format', type=str, dest='vis_format')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    assert args.vis_format, "Please set vis_format."
    return args

kpt = {
        'num': 133,
        'name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', # body
            *['Face_' + str(i) for i in range(52,69)], # face contour
            *['Face_' + str(i) for i in range(1,52)], # face
            'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand
            'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4') # right hand
        }
exclude_hand_idxs = [i for i in list(range(kpt['name'].index('L_Wrist_Hand'), kpt['name'].index('R_Pinky_4')+1)) if kpt['name'][i][-2:] in ['_2', '_3', '_4']]

# get path
args = parse_args()
root_path = args.root_path
out_path = osp.join(root_path, 'masks')
os.makedirs(out_path, exist_ok=True)
vis_format = args.vis_format

# load SAM 
ckpt_path = './sam_vit_h_4b8939.pth'
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=ckpt_path).cuda()
predictor = SamPredictor(sam)

# run SAM
img_path_list = glob(osp.join(root_path, 'images', '*.png'))
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]
if vis_format == 'video':
    video_save = cv2.VideoWriter(osp.join(root_path, 'masks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
for frame_idx in tqdm(frame_idx_list):
    
    # load image
    img_path = osp.join(root_path, 'images', str(frame_idx) + '.png')
    img = cv2.imread(img_path)

    # load keypoints
    kpt_path = osp.join(root_path, 'keypoints_whole_body', str(frame_idx) + '.json')
    with open(kpt_path) as f:
        kpt = np.array(json.load(f), dtype=np.float32)
    kpt = np.stack([kpt[i] for i in range(len(kpt)) if i not in exclude_hand_idxs])
    kpt = kpt[kpt[:,2] > 0.2,:2]
    bbox = get_bbox(kpt, np.ones_like(kpt[:,0]))
    bbox[2:] += bbox[:2] # xywh -> xyxy

    # use keypoints as prompts
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_input)
    masks, scores, logits = predictor.predict(point_coords=kpt, point_labels=np.ones_like(kpt[:,0]), box=bbox[None,:], multimask_output=False)
    mask_input = logits[np.argmax(scores), :, :]
    masks, _, _ = predictor.predict(point_coords=kpt, point_labels=np.ones_like(kpt[:,0]), box=bbox[None,:], multimask_output=False, mask_input=mask_input[None])
    mask = masks.sum(0) > 0
    
    # save mask and visualization
    if vis_format == 'video':
        cv2.imwrite(osp.join(out_path, str(frame_idx) + '.png'), mask * 255)
        img_masked = img.copy()
        img_masked[~mask] = 0
        frame = np.concatenate((img, img_masked),1)
        frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
        video_save.write(frame.astype(np.uint8))
    elif vis_format == 'image':
        cv2.imwrite(osp.join(out_path, str(frame_idx) + '.png'), mask * 255)
        img_masked = img.copy()
        img_masked[~mask] = 0
        cv2.imwrite(osp.join(out_path, str(frame_idx) + '_vis.png'), img_masked)


