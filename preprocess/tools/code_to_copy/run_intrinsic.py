import os
import os.path as osp
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from chrislib.general import view, tile_imgs, view_scale, uninvert
from chrislib.data_util import load_image
from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path
save_path = osp.join(root_path, 'image_intrinsics')
os.makedirs(save_path, exist_ok=True)

# load the models from the given paths
models = load_models('final_weights.pt')

# load an image
img_path_list = glob(osp.join(root_path, 'images', '*.png'))
for img_path in tqdm(img_path_list):
    frame_idx = img_path.split('/')[-1][:-4]
    image = load_image(img_path)

    # run the model on the image using R_0 resizing
    results = run_pipeline(
        models,
        image,
        resize_conf=0.0,
        maintain_size=True
    )

    # save albedo
    albedo = (results['albedo'] ** (1/2.2)).clip(0,1)
    cv2.imwrite(osp.join(save_path, frame_idx + '_albedo.png'), albedo[:,:,::-1]*255)
