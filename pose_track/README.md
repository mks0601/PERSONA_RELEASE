# Tracking 3D poses from images or a video

* This code includes **pose tracking function from images or a video**.

## Directory
You can follow one of below two directories.
```
${ROOT}
|-- main
|-- data
|-- |-- subjects
|-- |-- |-- $SUBJECT_NAME
|-- |-- |-- |-- images
|-- |-- |-- |-- |-- 0.png
|-- |-- |-- |-- |-- 1.png
|-- |-- |-- |-- |-- ...
|-- |-- |-- |-- |-- N.png
```
```
${ROOT}
|-- main
|-- data
|-- |-- subjects
|-- |-- |-- $SUBJECT_NAME
|-- |-- |-- |-- video.mp4
|-- |-- |-- |-- valid_frame_list.txt
```
* If you want only partial frames from videos, you can optionally save `valid_frame_list.txt` with your desired frame indices with newline for each frame index. For example, if you want 0th, 1st, 2nd, and 100th frames, `valid_frame_list.txt` should include `0\n1\n2\n100\n`. This is used only for the slow but accurate version.

## Start (fast but less accurate version)
* It 1) runs pre-trained SMPL-X regressor and 2) optionally temrapolly smooth regressed ones.
* Go to `tools/SMPLest-X` folder.
* Run `python run_smplest_x.py --root_path $PATH --data_format $DATA_FORMAT`.
* `$PATH` is an **absolute path**. An example of `$PATH` of above directory is `$ROOT/data/subjects/$SUBJECT_NAME`.
* `$DATA_FORMAT` is one of `image` or `video`. If you follow the first directoty, set `$DATA_FORMAT` to `image`. Otherwise, set `$DATA_FORMAT` to `video`.
* Tracking results are saved in `$ROOT/data/subjects/$SUBJECT_NAME`.

## Start (slow but accurate version)
* It 1) fits regressed SMPL-X meshes to estimated 2D keypoints and depthmaps and 2) temporally smooth them when `$DATA_FORMAT` is `video`.
* Go to `tools` folder.
* Run `python run.py --root_path $PATH --data_format $DATA_FORMAT`.
* `$PATH` is an **absolute path**. An example of `$PATH` of above directory is `$ROOT/data/subjects/$SUBJECT_NAME`.
* `$DATA_FORMAT` is one of `image` or `video`. If you follow the first directoty, set `$DATA_FORMAT` to `image`. Otherwise, set `$DATA_FORMAT` to `video`.
* Tracking results are saved in `$ROOT/data/subjects/$SUBJECT_NAME`.

## Tracked poses examples
* [Download](https://drive.google.com/drive/folders/1uSdCSUAihk96iaXnGPyjvAzpZWyZNXFs?usp=sharing)

## Reference
```
@inproceedings{sim2025persona,
  title={{PERSONA}: Personalized Whole-Body {3D} Avatar with Pose-Driven Deformations from a Single Image},
  author = {Sim, Geonhee and Moon, Gyeongsik},  
  booktitle={ICCV},
  year={2025}
}
```
