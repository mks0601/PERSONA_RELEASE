import os
import os.path as osp
from glob import glob
import cv2
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--data_format', type=str)
    args = parser.parse_args()
    assert args.root_path
    assert args.data_format in ['image', 'video']
    return args

args = parse_args()
root_path, data_format = args.root_path, args.data_format
cur_path = osp.dirname(osp.abspath(__file__))

# extract frames
if data_format == 'video':
    os.makedirs(osp.join(root_path, 'images'), exist_ok=True)
    vidcap = cv2.VideoCapture(osp.join(root_path, 'video.mp4'))
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = vidcap.read()
    frame_idx = 0
    while success:
        print('Extracting frames from video.mp4... ' + str(frame_idx) + '/' + str(frame_num), end='\r')
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

# run tracker (track human from a video)
if data_format == 'video':
    cmd = 'python track_human.py --root_path ' + root_path
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when tracking human. terminate the script.')
        sys.exit()

# run smplest-x (get smplx parameters)
os.chdir(osp.join(cur_path, 'SMPLest-X'))
cmd = 'python main/inference.py --root_path ' + root_path + ' --data_format ' + data_format
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running SMPLest-X. terminate the script.')
    sys.exit()
os.chdir(cur_path)

# run DECA (replace face smplx parameters with those from smplest-x)
os.chdir(osp.join(cur_path, 'DECA'))
cmd = 'python run_deca_pose_track.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running DECA. terminate the script.')
    sys.exit()
os.chdir(cur_path)

# run hand4whole (replace hand smplx parameters with those from smplest-x)
os.chdir(osp.join(cur_path, 'Hand4Whole_RELEASE/demo'))
cmd = 'python run_hand4whole.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running Hand4Whole. terminate the script.')
    sys.exit()
os.chdir(cur_path)
cmd = 'python smooth_smplx_params.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when smoothing outputs of Hand4Whole. terminate the script.')
    sys.exit()
os.chdir(cur_path)

# visualize smplx
cmd = 'python vis.py --root_path ' + root_path + ' --data_format ' + data_format
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running visualizing smplx meshes. terminate the script.')
    sys.exit()

# get background video
cmd = 'python get_bkg_video.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when getting background video. terminate the script.')
    sys.exit()
