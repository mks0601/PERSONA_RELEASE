import os
import os.path as osp
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
root_path = osp.join(root_path, 'captured')

# DECA (get initial FLAME parameters)
os.chdir(osp.join(cur_path, 'DECA'))
cmd = 'python run_deca.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running DECA (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# SMPLest-X (get initial SMPLX parameters)
os.chdir(osp.join(cur_path, 'SMPLest-X'))
cmd = 'python main/inference.py --root_path ' + root_path + ' --data_format image' 
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running SMPLest-X. terminate the script.')
    sys.exit()
os.system('mv ' + osp.join(root_path, 'smplx') + ' ' + osp.join(root_path, 'smplx_init'))
os.system('mv ' + osp.join(root_path, 'smplx_init', 'params', '*') + ' ' + osp.join(root_path, 'smplx_init', '.'))
os.system('rm -rf ' + osp.join(root_path, 'smplx_init', 'params'))
os.chdir(cur_path)

# sapiens (keypoints)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_kpt.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens pose (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# segment-anything (binary mask using keypoints from sapiens)
os.chdir(osp.join(cur_path, 'segment-anything'))
cmd = 'python run_sam.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running segment-anything (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (depth using binary masks from Segment Anything)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_depth.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens depth (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (normal using binary masks from Segment Anything)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_normal.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens normal (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (part segmentation using binary masks from Segment Anything)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_seg.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens segmentation (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# Intrinsic (albedo)
os.chdir(osp.join(cur_path, 'Intrinsic'))
cmd = 'python run_intrinsic.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running Intrinsic (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# ResShift (face patch super-resolution)
os.chdir(osp.join(cur_path, 'ResShift'))
cmd = 'python run_resshift.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running ResShift (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (face patch keypoints)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_kpt.py --root_path ' + osp.join(root_path, 'face_patches') + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens face pose (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# fit SMPLX to 'captured' images, render mesh, and unwrap texture
os.chdir(osp.join(cur_path, '..', 'main'))
os.makedirs(osp.join('..', 'data', 'subjects'), exist_ok=True)
if not osp.isdir(osp.join('..', 'data', 'subjects', subject_id)):
    os.system('ln -s ' + root_path + ' ' + osp.join('..', 'data', 'subjects', subject_id))
cmd = 'python fit_w_id_opt.py --subject_id ' + subject_id
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when fitting (run_captured.py). terminate the script.')
    sys.exit()
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, 'smplx_optimized') + ' ' + osp.join(root_path, '.')
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when moving fitting (run_captured.py). terminate the script.')
    sys.exit()
cmd = 'python vis.py --subject_id ' + subject_id + ' --split captured'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when visualizing fitting (run_captured.py). terminate the script.')
    sys.exit()
cmd = 'python unwrap.py --subject_id ' + subject_id + ' --split captured'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when unwrapping texture from fitting (run_captured.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

