import os
import os.path as osp
import argparse
import numpy as np
import cv2
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True)
    args = parser.parse_args()
    root_path = args.root_path

    # load video
    video_cap = cv2.VideoCapture(osp.join(root_path, 'video.mp4'))

    # prepare save
    img_height, img_width = cv2.imread(glob(osp.join(root_path, 'images', '*.png'))[0]).shape[:2]
    video_save = cv2.VideoWriter(osp.join(root_path, 'video_bkg.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width, img_height))
    frame_idx = 0
    save_path = osp.join(root_path, 'images_bkg')
    os.makedirs(save_path, exist_ok=True)

    # inpaint video
    while True:
        ret, frame = video_cap.read()

        mask_path = osp.join(root_path, 'masks', str(frame_idx) + '.png')
        if osp.isfile(mask_path):
            mask = cv2.imread(mask_path)[:,:,0]
            do_inpaint = (mask > 0).astype(np.uint8)
            frame = cv2.inpaint(frame, do_inpaint, 3, cv2.INPAINT_TELEA)
            cv2.imwrite(osp.join(save_path, str(frame_idx) + '.png'), frame)
            video_save.write(frame.astype(np.uint8))

        frame_idx += 1
        if not ret:
            break

if __name__ == '__main__':
    main()

