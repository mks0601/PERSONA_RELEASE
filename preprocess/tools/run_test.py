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
root_path = osp.join(root_path, 'test')

# DECA (get initial FLAME parameters)
os.chdir(osp.join(cur_path, 'DECA'))
cmd = 'python run_deca.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running DECA (run_test.py). terminate the script.')
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
    print('something bad happened when running sapiens pose (run_test.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# segment-anything (binary mask using keypoints from sapiens)
os.chdir(osp.join(cur_path, 'segment-anything'))
cmd = 'python run_sam.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running segment-anything (run_test.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (depth using binary masks from Segment Anything)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_depth.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens depth (run_test.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (normal using binary masks from Segment Anything)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_normal.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens normal (run_test.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# sapiens (part segmentation using binary masks from Segment Anything)
os.chdir(osp.join(cur_path, 'sapiens/lite/scripts/demo/torchscript'))
cmd = 'python run_sapiens_seg.py --root_path ' + root_path + ' --vis_format image'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when running sapiens segmentation (run_test.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

# fit SMPLX to 'test' images, render mesh, and unwrap texture (for the ID parameters, load the optimized ones and do not optimize them)
os.chdir(osp.join(cur_path, '..', 'main'))
cmd = 'python fit_wo_id_opt.py --subject_id ' + subject_id
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when fitting (run_test.py). terminate the script.')
    sys.exit()
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, 'smplx_optimized') + ' ' + osp.join(root_path, '.')
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when moving fitting (run_test.py). terminate the script.')
    sys.exit()
cmd = 'python vis.py --subject_id ' + subject_id + ' --split test'
print(cmd)
result = os.system(cmd)
if (result != 0):
    print('something bad happened when visualizing fitting (run_test.py). terminate the script.')
    sys.exit()
os.chdir(cur_path)

