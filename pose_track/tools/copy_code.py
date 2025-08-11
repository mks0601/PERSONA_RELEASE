import os
import os.path as osp

src_path = './code_to_copy'

# torchgeometry
import torchgeometry
import pathlib
orig_path = pathlib.Path(torchgeometry.__file__).parent
cmd = 'cp ' + osp.join(src_path, 'torchgeometry', 'core', 'conversions.py') + ' ' + osp.join(orig_path, 'core', 'conversions.py')
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

# SMPLest-X
cmd = 'cp ' + osp.join(src_path, 'run_smplest_x.py') + ' ./SMPLest-X/main/.'
os.system(cmd)

