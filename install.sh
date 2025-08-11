#!/bin/bash

# Move into third_modules directory
pushd third_modules

# ===== 1. DECA =====
echo "Downloading DECA..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-484bcf9ca39ac87ddd0d/DECA.zip
unzip -o DECA.zip
rm -f DECA.zip

# ===== 2. Hand4Whole =====
echo "Downloading Hand4Whole..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-484bcf9ca39ac87ddd0d/Hand4Whole_RELEASE.zip
unzip -o Hand4Whole_RELEASE.zip
rm -f Hand4Whole_RELEASE.zip

# ===== 3. Intrinsic =====
echo "Downloading Intrinsic..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-484bcf9ca39ac87ddd0d/Intrinsic.zip
unzip -o Intrinsic.zip
rm -f Intrinsic.zip
pushd Intrinsic
python setup.py install
popd  # Return to third_modules

# ===== 4. ResShift =====
echo "Downloading ResShift..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-484bcf9ca39ac87ddd0d/ResShift.zip
unzip -o ResShift.zip
rm -f ResShift.zip

# ===== 5. mip-splatting =====
echo "Downloading mip-splatting..."
git clone https://github.com/autonomousvision/mip-splatting
pushd mip-splatting/submodules/diff-gaussian-rasterization
python setup.py install
popd  # Return to third_modules

# ===== 6. MimicMotion =====
echo "Downloading MimicMotion..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-484bcf9ca39ac87ddd0d/ResShift.zip
unzip -o MimicMotion.zip
rm -f MimicMotion.zip
mkdir -p models/DWPose
wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
wget -P models/ https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-7d04ffac1bfb95ed7960/motion_0.zip
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-7d04ffac1bfb95ed7960/motion_1.zip
mv motion_0.zip ../preprocess/tools/code_to_copy/MimicMotion/.
mv motion_1.zip ../preprocess/tools/code_to_copy/MimicMotion/.
pushd ../preprocess/tools/code_to_copy/MimicMotion/
unzip motion_0.zip
unzip motion_1.zip
popd  # Return to third_modules

# ===== 7. segment-anything =====
echo "Downloading segment anything..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-7d04ffac1bfb95ed7960/segment-anything.zip
unzip -o segment-anything.zip
rm -f segment-anything.zip
pushd segment-anything
python setup.py install
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
popd  # Return to third_modules

# ===== 8. sam2 =====
echo "Downloading sam2..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-7d04ffac1bfb95ed7960/sam2.zip
unzip -o sam2.zip
rm -f sam2.zip
pushd sam2
python setup.py install
pushd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
popd
popd  # Return to third_modules

# ===== 9. sapiens =====
echo "Downloading sapiens..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-7d04ffac1bfb95ed7960/sapiens.zip
unzip -o sapiens.zip
rm -f sapiens.zip
pushd sapiens/lite/scripts/demo/torchscript/checkpoints
wget https://huggingface.co/facebook/sapiens-pose-bbox-detector/resolve/c844c2df76f1613d7c5e2910d8bf30039a55a386/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth # human detection
wget https://huggingface.co/facebook/sapiens-depth-1b-torchscript/resolve/main/sapiens_1b_render_people_epoch_88_torchscript.pt2 # depth
wget https://huggingface.co/facebook/sapiens-normal-1b-torchscript/resolve/main/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2 # normal
wget https://huggingface.co/noahcao/sapiens-pose-coco/resolve/main/sapiens_host/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727.pth # pose
wget https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 # seg
popd
popd  # Return to third_modules

# ===== 10. SMPLest-X =====
echo "Downloading smplest-x..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/untagged-7d04ffac1bfb95ed7960/SMPLest-X.zip
unzip -o SMPLest-X.zip
rm -f SMPLest-X.zip
pushd SMPLest-X/pretrained_models/smplest_x_h
wget https://huggingface.co/waanqii/SMPLest-X/resolve/main/smplest_x_h.pth.tar
popd
popd  # Return to third_modules

# ===== 11. Move YOLO model file =====
echo "Downloading YOLO model..."
wget https://huggingface.co/Ultralytics/YOLO11/resolve/d3043e98a1ad0e2956728c13cf1e041e0fa4220f/yolo11l-pose.pt
mv yolo11l-pose.pt ../preprocess/tools/.

# ===== 12. human model files =====
echo "Downloading human model files..."
gdown 1kk5NyLurez9Dud5_d11aeMipMZsS5X2K
unzip -o human_model_files.zip
rm -f human_model_files.zip

