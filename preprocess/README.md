# Preprocessing to create avatars

* This branch provides code to **preprocess a single image to create avatars**.

## Start
```
${ROOT}
|-- main
|-- data
|-- |-- subjects
|-- |-- |-- $SUBJECT_NAME
|-- |-- |-- |-- captured
|-- |-- |-- |-- |-- images
|-- |-- |-- |-- |-- |-- 0.png
```
* Put your single image to `$ROOT/data/subjects/$SUBJECT_NAME/captured/images/0.png`.
* For NeuMan and X-Humans datasets, you can use `tools/NeuMan/prepare_captured_test_imgs.py` and `tools/XHumans/prepare_captured_test_imgs.py`, respectively, to prepare `captured` and `test` images.

## 1. Preprocess `captured` data (necessary to create avatars)
* Go to `tools` folder.
* Run `export CUDA_VISIBLE_DEVICES=$GPU_ID` where `$GPU_ID` is your desired GPU index.
* Run `python run_captured.py --root_path $ROOT_PATH`, where `$ROOT_PATH` is an **absolute path** to the subject. In the case of above directory, `$ROOT_PATH` is `$ROOT/data/subjects/$SUBJECT_NAME`.

## 2. Preprocess `generated` data (necessary to create avatars)
* Go to `tools` folder.
* Run `export CUDA_VISIBLE_DEVICES=$GPU_ID` where `$GPU_ID` is your desired GPU index. 
* Run `python run_generated.py --root_path $ROOT_PATH`, where `$ROOT_PATH` is an **absolute path** to the subject. In the case of above directory, `$ROOT_PATH` is `$ROOT/data/subjects/$SUBJECT_NAME`.

## 3. Preprocess `test` data (only for quantitative evaluations of NeuMan and X-Humans datasets)
* Download original dataset from [NeuMan](https://github.com/apple/ml-neuman?tab=readme-ov-file) and [X-Humans](https://skype-line.github.io/projects/X-Avatar/).
* Official training/testing split files of `NeuMan` dataset is available in [here](https://drive.google.com/drive/folders/1QJztYKjI9tC90U6mELwt-RF1V09LBqjW?usp=sharing). Save this to each subject folder of the original dataset.
* Run `export CUDA_VISIBLE_DEVICES=$GPU_ID` where `$GPU_ID` is your desired GPU index.
* Go to `tools/NeuMan` or `tools/XHumans` folder and run `python prepare_captured_test_imgs.py --root_path $DB_PATH --save_path $ROOT/data/subjects/$SUBJECT_NAME --split test` where `$DB_PATH` represents the original dataset path.
* Then, you can get below directory.
```
${ROOT}
|-- main
|-- data
|-- |-- subjects
|-- |-- |-- $SUBJECT_NAME
|-- |-- |-- |-- captured
|-- |-- |-- |-- generated_0
|-- |-- |-- |-- generated_1
|-- |-- |-- |-- test
|-- |-- |-- |-- |-- frame_idx_orig.txt
|-- |-- |-- |-- |-- images
|-- |-- |-- |-- |-- |-- 0.png
|-- |-- |-- |-- |-- |-- 0_bbox_orig.json
|-- |-- |-- |-- |-- |-- 1.png
|-- |-- |-- |-- |-- |-- 1_bbox_orig.json
|-- |-- |-- |-- |-- |-- ...
|-- |-- |-- |-- |-- |-- N.png
|-- |-- |-- |-- |-- |-- N_bbox_orig.json
|-- |-- |-- |-- |-- |-- img_shape_orig.json
```
* Go to `tools` folder.
* Run `python run_test.py --root_path $ROOT_PATH`, where `$ROOT_PATH` is an **absolute path** to the subject. In the case of above directory, `$ROOT_PATH` is `$ROOT/data/subjects/$SUBJECT_NAME`.

## Preprocess examples
* We provide an example of preprocessed data in below.
* [Gyeongsik](https://1drv.ms/u/c/3971ed61d3817b59/IQCoWfDnD27AR5qFYJc1REe0Ab6TXIDGuK6crE1EaIGiaVE?e=cD84f8).
* [Loose clothing 1](https://1drv.ms/u/c/3971ed61d3817b59/IQAHbQgEcsaUT6rx7JDaQ593ASSbKA_l-w4eiyda4vX-Cyo?e=JOwe2i).
* [Loose_clothing_2](https://1drv.ms/u/c/3971ed61d3817b59/IQBI84D0Xd8qQIgOo8yUAP5YAfDriy1rRGvioCbSEwHNGo4?e=Z2VB3W).
* [`captured` of NeuMan dataset](https://1drv.ms/f/c/3971ed61d3817b59/IgCIZ1E8DzthTaIYqe7n_fZdAbP7_SlMZS_NKaosO61cuIU?e=qFyVbv).
* [`captured` of X-Humans dataset](https://1drv.ms/f/c/3971ed61d3817b59/IgBDtPHUCaxpRLDzAvRSQg81AVqtJqZEj13p-4-PQwH0mOY?e=1ofpM1).

## Reference
```
@inproceedings{sim2025persona,
  title={{PERSONA}: Personalized Whole-Body {3D} Avatar with Pose-Driven Deformations from a Single Image},
  author = {Sim, Geonhee and Moon, Gyeongsik},  
  booktitle={ICCV},
  year={2025}
}
```
