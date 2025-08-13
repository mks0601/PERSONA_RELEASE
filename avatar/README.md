# Creating an avatar from a single image

* This code includes **avatar creation pipeline and animation function**.

## Directory
```
${ROOT}
|-- main
|-- data
|-- |-- subjects
|-- |-- |-- $SUBJECT_NAME
```
* `subjects` folder contains preprocessed data from [../preprocess](../preprocess/).

## Prepare
* Go to `tools/diffused_skinning_weights` and run `python make_skinning_weight.py` to prepare diffused skinning weights.
* It will output `diffused_skinning_weights.npy`, `skinning_grid_coords_x.npy`, `skinning_grid_coords_y.npy`, and `skinning_grid_coords_z.npy`.

## Train
* Go to `main` folder and run `python train.py --subject_id $SUBJECT_ID`. The checkpoints are saved in `output/model/$SUBJECT_ID`.
* You can see reconstruction results on the training frames by running `python test.py --subject_id $SUBJECT_ID --test_epoch 4`. The results are saved to `output/result/$SUBJECT_ID`.

## Visualize a rotating avatar with the neutral pose
* Go to `main` folder and run `python get_neutral_pose.py --subject_id $SUBJECT_ID --test_epoch 4`.
* You can see a rotating avatar with the neutral pose in `./main/neutral_pose`.

## Animation
* Go to `main` folder and run `python animation.py --subject_id $SUBJECT_ID --test_epoch 4 --motion_path $PATH` if you want to use an avatar in `output/model_dump/$SUBJECT_ID`. `$PATH` should contain SMPL-X parameters to animate the avatar. You can prepare `$PATH` with [../pose_track](../pose_track).
* We provide SMPL-X parameters of several videos (examples of `$PATH`) in [here](https://drive.google.com/drive/folders/1uSdCSUAihk96iaXnGPyjvAzpZWyZNXFs?usp=sharing).
* You can enable `--use_bkg` option to use background frames. To obtain bakground frames, go to [../pose_track](../pose_track)

## Test and evaluation (NeuMan and X-Humans datasets)
* For the evaluation, we optimize SMPL-X paraemeters of testing frames with image loss while fixing the pre-trained avatars following [1](https://github.com/aipixel/GaussianAvatar/issues/14), [2](https://github.com/mikeqzy/3dgs-avatar-release/issues/21), and [Section 4 B Evaluation](https://arxiv.org/pdf/2106.13629). This is specified in [here]().
* Go to `tools` folder and run `python prepare_fit_pose_to_test.py --root_path ../output/model_dump/$SUBJECT_ID` if you want to use an avatar in `output/model_dump/$SUBJECT_ID`. It simply sets `epoch` of a checkpoint to 0 and save it to `'output/model_dump/$SUBJECT_ID' + '_fit_pose_to_test'`.
* Go to `main` folder and run `python train.py --subject_id $SUBJECT_ID --fit_pose_to_test --continue`.
* You can see test results on the testing frames by running `python test.py --subject_id $SUBJECT_ID --fit_pose_to_test --test_epoch 4`. The results are saved to `'output/result/$SUBJECT_ID' + '_fit_pose_to_test'`.
* For the evaluation of the NeuMan dataset, go to `tools` folder and run `python eval_neuman.py --output_path '../output/result/$SUBJECT_ID' + '_fit_pose_to_test' --subject_id $SUBJECT_ID`.
* For the evaluation of the X-Humans dataset, go to `tools` folder and run `python eval_xhumans.py --output_path '../output/result/$SUBJECT_ID' + '_fit_pose_to_test' --subject_id $SUBJECT_ID`.

## Pre-trained avatars
* [Download](https://drive.google.com/drive/folders/1J0z0HBEYB03r9svgpeLO2AqAvV1YAzRk?usp=sharing)

## Reference
```
@inproceedings{sim2025persona,
  title={{PERSONA}: Personalized Whole-Body {3D} Avatar with Pose-Driven Deformations from a Single Image},
  author = {Sim, Geonhee and Moon, Gyeongsik},  
  booktitle={ICCV},
  year={2025}
}
```
