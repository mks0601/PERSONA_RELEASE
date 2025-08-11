import os
import os.path as osp
import argparse
import torch
import numpy as np
import cv2
import sys
sys.path.insert(0, './segment-anything')
sys.path.insert(0, './sam2')
from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
from ultralytics import YOLO

def sample_points_mask_unified(mask, n_points=10):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    idxs = np.linspace(0, len(coords) - 1, n_points, dtype=int)
    points = coords[idxs][:, [1, 0]].astype(np.float32)  # (x, y)
    labels = np.ones(len(points), dtype=np.int32)
    return points, labels

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

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

def save_video(frames, path, fps=30):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.shape[2] == 3:  # RGB → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()

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
            mask = (logits[0] > 0).cpu().numpy()[0]
            frame_masks.append((mask * 255).astype(np.uint8))
        tracker.reset_state(state)
    return frame_masks

def inpaint_video(frames, masks):
    bkg_frames = []
    for i in range(len(frames)):
        frame, mask = frames[i], masks[i]
        do_inpaint = (mask > 0).astype(np.uint8)
        frame = cv2.inpaint(frame, do_inpaint, 3, cv2.INPAINT_TELEA)
        bkg_frames.append(frame)
    return bkg_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    print('Loading video')
    frames = load_video(args.input_path)
    
    print('Detecting human from the first frame')
    yolo_ckpt = './yolo11l-pose.pt'
    bbox = get_bbox(yolo_ckpt, frames[0])

    print('Extracting person mask from first frame')
    sam_ckpt = './segment-anything/sam_vit_h_4b8939.pth'
    mask = extract_person_mask(frames[0], sam_ckpt, bbox)

    print('Tracking person mask')
    sam2_ckpt = './sam2/checkpoints/sam2.1_hiera_large.pt'
    sam2_cfg = 'sam2.1_hiera_l.yaml'
    GlobalHydra.instance().clear()
    initialize(config_path='sam2/sam2/configs/sam2.1', job_name="sam2_inference", version_base=None)
    tracker = build_sam2_video_predictor(sam2_cfg, sam2_ckpt)
    masks = track_mask(args.input_path, mask, sam2_ckpt, sam2_cfg)
    
    print('Inpaint video')
    bkg_frames = inpaint_video(frames, masks)

    save_path = osp.join(args.output_path, 'bkg.mp4')
    save_video(bkg_frames, save_path)
    print(f"[Done] Saved bkg.mp4 to {save_path}")

if __name__ == '__main__':
    main()

