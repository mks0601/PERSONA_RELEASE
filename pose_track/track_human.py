import sys
sys.path.insert(0, './segment-anything')
sys.path.insert(0, './sam2')

import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from glob import glob
import json
from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
from torchvision.ops import masks_to_boxes

def sample_points_mask_unified(mask, n_points=10):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    idxs = np.linspace(0, len(coords) - 1, n_points, dtype=int)
    points = coords[idxs][:, [1, 0]].astype(np.float32)  # (x, y)
    labels = np.ones(len(points), dtype=np.int32)
    return points, labels

def get_bbox(yolo_ckpt, frame):
    model = YOLO(yolo_ckpt) 
    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]
    bboxes = results.boxes.xyxy.cpu().numpy()       # [N, 4]
    keypoints = results.keypoints.xyn.cpu().numpy()  # [N, 17, 2] keypoints are in [0,1] range
    def is_full_body(kps):
        needed = kps[[13, 14, 15, 16]]
        return np.all((needed > 0) & (needed < 1))
    full_body_indices = [i for i, kp in enumerate(keypoints) if is_full_body(kp)]
    if len(full_body_indices) == 0:
        areas = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
        bbox = bboxes[np.argmax(areas)].astype(int)
    else:
        areas = [(bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1]) for i in full_body_indices]
        best_idx = full_body_indices[np.argmax(areas)]
        bbox = bboxes[best_idx].astype(int)
    return bbox

def extract_person_mask(frame, sam_checkpoint, input_box, model_type='vit_h', device='cuda'):
    model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(model)
    predictor.set_image(frame)
    input_box = np.array(input_box)
    masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
    predictor.reset_image()
    return masks[0].astype(np.uint8)

def track_mask(frames, init_mask, sam2_ckpt, sam2_cfg):
    tracker = build_sam2_video_predictor(sam2_cfg, sam2_ckpt)
    with torch.inference_mode(), torch.autocast("cuda"):
        state = tracker.init_state(frames)
        points, labels = sample_points_mask_unified(init_mask)
        tracker.add_new_points(state, frame_idx=0, obj_id=1, points=points, labels=labels)
        frame_masks = []
        for _, _, logits in tracker.propagate_in_video(state):
            frame_masks.append((logits[0] > 0)[0])
        tracker.reset_state(state)
    frame_masks = torch.stack(frame_masks).float()
    return frame_masks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str)
    args = parser.parse_args()
    assert args.root_path
    return args

def main():
    args = parse_args()

    root_path = args.root_path
    img_path_list = glob(osp.join(root_path, 'images', '*'))
    frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
    img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]

    # detect human from the first frame
    init_frame = cv2.imread(osp.join(root_path, 'images', str(frame_idx_list[0]) + '.png'))[:,:,::-1]
    yolo_ckpt = './yolo11l-pose.pt'
    init_bbox_path = osp.join(root_path, 'bboxes', str(frame_idx_list[0]) + '.json')
    if osp.isfile(init_bbox_path): # if we have initial bbox, use it
        with open(init_bbox_path) as f:
            bbox = np.array(json.load(f), dtype=np.float32)
        bbox[2] += bbox[0]; bbox[3] += bbox[1]; # xywh -> xyxy
    else:
        bbox = get_bbox(yolo_ckpt, init_frame)
    sam_ckpt = './segment-anything/sam_vit_h_4b8939.pth'
    mask = extract_person_mask(init_frame, sam_ckpt, bbox)

    # track human from the video
    sam2_ckpt = './sam2/checkpoints/sam2.1_hiera_large.pt'
    sam2_cfg = 'sam2.1_hiera_l.yaml'
    GlobalHydra.instance().clear()
    initialize(config_path='sam2/sam2/configs/sam2.1', job_name="sam2_inference", version_base=None)
    tracker = build_sam2_video_predictor(sam2_cfg, sam2_ckpt)
    masks = track_mask(osp.join(root_path, 'video.mp4'), mask, sam2_ckpt, sam2_cfg)

    video_bbox_save = cv2.VideoWriter(osp.join(root_path, 'bboxes.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
    video_mask_save = cv2.VideoWriter(osp.join(root_path, 'masks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))
    for i, frame_idx in enumerate(tqdm(frame_idx_list)):
        
        # prepare input image
        img_path = osp.join(root_path, 'images', str(frame_idx) + '.png')
        img = cv2.imread(img_path)
        
        # get bbox from mask
        if masks[i].sum() == 0:
            continue
        bbox = masks_to_boxes(masks[i][None])[0]
        bbox[2:] -= bbox[:2] # xyxy -> xywh
        bbox = bbox.detach().cpu().numpy()
        img_bbox = cv2.rectangle(img.copy(), (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0,255,0), 2)
        mask = masks[i].detach().cpu().numpy()
        img_masked = img * mask[:,:,None]

        # write frame
        out = np.concatenate((img, img_bbox),1).astype(np.uint8)
        out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.025), int(out.shape[0]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, 2)
        video_bbox_save.write(out)
        out = np.concatenate((img, img_masked),1).astype(np.uint8)
        out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.025), int(out.shape[0]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, 2)
        video_mask_save.write(out)

        # save
        save_path = osp.join(root_path, 'bboxes')
        os.makedirs(save_path, exist_ok=True)
        with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
            json.dump(bbox.tolist(), f)
        save_path = osp.join(root_path, 'masks')
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(osp.join(save_path, str(frame_idx) + '.png'), mask*255)

if __name__ == "__main__":
    main()
