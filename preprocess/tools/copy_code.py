import os
import os.path as osp

src_path = './code_to_copy'

# torchgeometry
import torchgeometry
import pathlib
orig_path = pathlib.Path(torchgeometry.__file__).parent
cmd = 'cp ' + osp.join(src_path, 'torchgeometry', 'core', 'conversions.py') + ' ' + osp.join(orig_path, 'core', 'conversions.py')
os.system(cmd)

# DECA
cmd = 'cp ' + osp.join(src_path, 'run_deca.py') + ' ./DECA/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'decalib', 'deca.py') + ' ./DECA/decalib/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'decalib', 'datasets', 'datasets.py') + ' ./DECA/decalib/datasets/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'DECA', 'demos', 'demo_reconstruct.py') + ' ./DECA/demos/.'
os.system(cmd)

# Hand4Whole
cmd = 'cp ' + osp.join(src_path, 'run_hand4whole.py') + ' ./Hand4Whole_RELEASE/demo/.'
os.system(cmd)

# sapiens
cmd = 'cp ' + osp.join(src_path, 'sapiens/lite/demo/*') + ' ./sapiens/lite/demo/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'sapiens/lite/scripts/demo/torchscript/*') + ' ./sapiens/lite/scripts/demo/torchscript/.'
os.system(cmd)

# Segment Anything
cmd = 'cp ' + osp.join(src_path, 'run_sam.py') + ' ./segment-anything/.'
os.system(cmd)

# Intrinsic
cmd = 'cp ' + osp.join(src_path, 'run_intrinsic.py') + ' ./Intrinsic/.'
os.system(cmd)

# ResShift
cmd = 'cp ' + osp.join(src_path, 'run_resshift.py') + ' ./ResShift/.'
os.system(cmd)

# MimicMotion
cmd = 'cp ' + osp.join(src_path, 'run_mimic_motion.py') + ' ./MimicMotion/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'MimicMotion', 'inference.py') + ' ./MimicMotion/.'
os.system(cmd)
cmd = 'cp ' + osp.join(src_path, 'MimicMotion', 'configs', 'test.yaml') + ' ./MimicMotion/configs/.'
os.system(cmd)


