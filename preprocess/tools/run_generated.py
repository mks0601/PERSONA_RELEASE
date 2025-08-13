import os
import os.path as osp
from glob import glob
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path
cur_path = osp.dirname(osp.abspath(__file__))
if root_path[-1] == '/':
    subject_id = root_path.split('/')[-2]
else:
    subject_id = root_path.split('/')[-1]

# iterate each motion
motion_names = sorted([x.split('/')[-1] for x in glob(osp.join(cur_path, 'prepare_training_video_generation', 'motion_*'))])
for motion_name in motion_names:

    if motion_name not in ['motion_0', 'motion_1']:
        continue

    split_name = motion_name.replace('motion', 'generated') # e.g., generated_0
    split_root_path = osp.join(root_path, split_name)
    os.makedirs(split_root_path, exist_ok=True)
    
    # MimicMotion
    os.chdir(osp.join(cur_path, 'prepare_training_video_generation'))
    cmd = 'python prepare_video.py --root_path ' + split_root_path + ' --motion_path ' + motion_name # hard-coded path to get motion
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when preparing video input for MimicMotion (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(osp.join(cur_path, 'MimicMotion'))
    cmd = 'python run_mimic_motion.py --root_path ' + split_root_path
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running MimicMotion (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)
    
    # sapiens (keypoints)
    os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
    cmd = 'python run_sapiens_kpt.py --root_path ' + split_root_path + ' --vis_format video'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running sapiens pose (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)

    # fix foot poses
    os.chdir(osp.join(cur_path, 'prepare_training_video_generation'))
    cmd = 'python fix_foot_pose.py --root_path ' + split_root_path
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when fixing foot poses (run_generated.py). terminate the script.')
        sys.exit()
    cmd = 'python vis.py --root_path ' + split_root_path
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when visualizing results of MimicMotion (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)
    
    # segment-anything (binary mask using keypoints from SMPL-X)
    os.chdir(osp.join(cur_path, 'segment-anything'))
    cmd = 'python run_sam.py --root_path ' + split_root_path + ' --vis_format video'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running segment-anything (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)

    # sapiens (depth using binary masks from segment-anything)
    os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
    cmd = 'python run_sapiens_depth.py --root_path ' + split_root_path + ' --vis_format video'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running sapiens depth (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)

    # sapiens (normal using binary masks from segment-anything)
    os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
    cmd = 'python run_sapiens_normal.py --root_path ' + split_root_path + ' --vis_format video'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running sapiens normal (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)

    # sapiens (part segmentation using binary masks from Segment Anything)
    os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
    cmd = 'python run_sapiens_seg.py --root_path ' + split_root_path + ' --vis_format video'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running sapiens segmentation (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)

    # unwrap texture
    os.chdir(osp.join(cur_path, '..', 'main'))
    os.makedirs(osp.join('..', 'data', 'subjects'), exist_ok=True)
    if not osp.isdir(osp.join('..', 'data', 'subjects', subject_id)):
        os.system('ln -s ' + root_path + ' ' + osp.join('..', 'data', 'subjects', subject_id))
    cmd = 'python unwrap.py --subject_id ' + subject_id + ' --split ' + split_name
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when unwrapping texture (run_generated.py). terminate the script.')
        sys.exit()
    os.chdir(cur_path)

