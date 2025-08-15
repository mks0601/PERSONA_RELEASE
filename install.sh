#!/bin/bash
set -e

# Move into third_modules directory
pushd third_modules

# ===== 1. DECA =====
echo "Downloading DECA..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/DECA.zip
unzip -o DECA.zip
rm -f DECA.zip

# ===== 2. Hand4Whole =====
echo "Downloading Hand4Whole..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/Hand4Whole_RELEASE.zip
unzip -o Hand4Whole_RELEASE.zip
rm -f Hand4Whole_RELEASE.zip

# ===== 3. Intrinsic =====
echo "Downloading Intrinsic..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/Intrinsic.zip
unzip -o Intrinsic.zip
rm -f Intrinsic.zip
pushd Intrinsic
python setup.py install
popd  # Return to third_modules

# ===== 4. ResShift =====
echo "Downloading ResShift..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/ResShift.zip
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
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/MimicMotion.zip
unzip -o MimicMotion.zip
rm -f MimicMotion.zip
mkdir -p models/DWPose
wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
wget -P models/ https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth
huggingface-cli download \
  stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --repo-type model \
  --revision main \
  --local-dir MimicMotion/models/SVD \
  --local-dir-use-symlinks False \
  --resume-download
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/motion_0.zip
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/motion_1.zip
mv motion_0.zip ../preprocess/tools/prepare_training_video_generation/.
mv motion_1.zip ../preprocess/tools/prepare_training_video_generation/.
pushd ../preprocess/tools/prepare_training_video_generation/
unzip motion_0.zip
unzip motion_1.zip
popd  # Return to third_modules

# ===== 7. segment-anything =====
echo "Downloading segment anything..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/segment-anything.zip
unzip -o segment-anything.zip
rm -f segment-anything.zip
pushd segment-anything
python setup.py install
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
popd  # Return to third_modules

# ===== 8. sam2 =====
echo "Downloading sam2..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/sam2.zip
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
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/sapiens.zip
unzip -o sapiens.zip
rm -f sapiens.zip
pushd sapiens/lite/scripts/demo/torchscript/checkpoints
wget https://huggingface.co/facebook/sapiens-pose-bbox-detector/resolve/c844c2df76f1613d7c5e2910d8bf30039a55a386/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth # human detection
wget https://huggingface.co/facebook/sapiens-depth-1b-torchscript/resolve/main/sapiens_1b_render_people_epoch_88_torchscript.pt2 # depth
wget https://huggingface.co/facebook/sapiens-normal-1b-torchscript/resolve/main/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2 # normal
wget https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 # seg
FILENAME="sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2" # pose
huggingface-cli download noahcao/sapiens-pose-coco \
  --include "sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/${FILENAME}" \
  --repo-type model \
  --local-dir ./tmp_hf_download \
  --local-dir-use-symlinks False \
  --resume-download
mv "./tmp_hf_download/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/${FILENAME}" "./${FILENAME}"
rm -rf ./tmp_hf_download
popd  # Return to third_modules

# ===== 10. SMPLest-X =====
echo "Downloading smplest-x..."
wget https://github.com/mks0601/PERSONA_RELEASE/releases/download/1.0/SMPLest-X.zip
unzip -o SMPLest-X.zip
rm -f SMPLest-X.zip
pushd SMPLest-X/pretrained_models/smplest_x_h
wget https://huggingface.co/waanqii/SMPLest-X/resolve/main/smplest_x_h.pth.tar
popd  # Return to third_modules

# ===== 11. human model files =====
echo "Downloading human model files..."
gdown 1kk5NyLurez9Dud5_d11aeMipMZsS5X2K
unzip -o human_model_files.zip
rm -f human_model_files.zip

# ===== 12. fix torchgeometry bug =====
ORIG_PATH=$(python -c "import torchgeometry, pathlib; print(pathlib.Path(torchgeometry.__file__).parent)")
cp "./torchgeometry_bug_fixed/core/conversions.py" "${ORIG_PATH}/core/conversions.py"
