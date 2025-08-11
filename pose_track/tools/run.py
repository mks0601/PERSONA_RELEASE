import os
import os.path as osp
import sys
from glob import glob
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--data_format', type=str, dest='data_format')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    assert args.data_format in ['image', 'video'], "Please set data_format among 'image' and 'video'."
    return args

# get path
args = parse_args()
root_path = args.root_path
cur_path = osp.dirname(osp.abspath(__file__))
if root_path[-1] == '/':
    subject_id = root_path.split('/')[-2]
else:
    subject_id = root_path.split('/')[-1]
data_format = args.data_format

# extract frames
if args.data_format == 'video':
    os.makedirs(osp.join(root_path, 'images'), exist_ok=True)
    vidcap = cv2.VideoCapture(osp.join(root_path, 'video.mp4'))
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = vidcap.read()
    frame_idx = 0
    while success:
        cv2.imwrite(osp.join(root_path, 'images', str(frame_idx) + '.png'), frame)
        success, frame = vidcap.read()
        frame_idx += 1

# remove unnecessary frames
if osp.isfile(osp.join(root_path, 'valid_frame_list.txt')):
    with open(osp.join(root_path, 'valid_frame_list.txt')) as f:
        frame_idx_list = [int(x) for x in f.readlines()]
    img_path_list = glob(osp.join(root_path, 'images', '*.png'))
    for img_path in img_path_list:
        frame_idx = int(img_path.split('/')[-1][:-4])
        if frame_idx not in frame_idx_list:
            cmd = 'rm ' + img_path
            os.system(cmd)

# make camera parameters
cmd = 'python make_virtual_cam_params.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when making virtual camera parameters. terminate the script.')
    sys.exit()

# Hand4Whole (get initial SMPLX parameters)
os.chdir(osp.join(cur_path, 'Hand4Whole_RELEASE/demo'))
cmd = 'python run_hand4whole.py --gpu 0 --root_path ' + root_path + ' --vis_format ' + data_format
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running Hand4Whole. terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (keypoints)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_kpt.py --root_path ' + root_path + ' --vis_format ' + data_format
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens pose. terminate the script.')
    sys.exit()
os.chdir(cur_path)

# segment-anything (binary mask using keypoints from sapiens)
os.chdir(osp.join(cur_path, 'segment-anything'))
cmd = 'python run_sam.py --root_path ' + root_path + ' --vis_format ' + data_format
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running segment-anything. terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (depth using binary masks from Segment Anything)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_depth.py --root_path ' + root_path + ' --vis_format ' + data_format
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens depth. terminate the script.')
    sys.exit()
os.chdir(cur_path)

# fit SMPLX to images and render mesh
os.chdir(osp.join(cur_path, '..', 'main'))
os.makedirs(osp.join('..', 'data', 'subjects'), exist_ok=True)
if not osp.isdir(osp.join('..', 'data', 'subjects', subject_id)):
    os.system('ln -s ' + root_path + ' ' + osp.join('..', 'data', 'subjects', subject_id))
cmd = 'python fit.py --subject_id ' + subject_id
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when fitting. terminate the script.')
    sys.exit()
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, 'smplx_optimized') + ' ' + osp.join(root_path, '.')
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when moving fitting. terminate the script.')
    sys.exit()
cmd = 'python vis.py --subject_id ' + subject_id + ' --vis_format ' + data_format
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when visualizing fitting. terminate the script.')
    sys.exit()
os.chdir(cur_path)

# smooth SMPLX
if data_format == 'video':
    cmd = 'python smooth_smplx_params.py --root_path ' + root_path
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when smoothing smplx parameters. terminate the script.')
        sys.exit()

