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
* Run `python run_captured.py --root_path $ROOT_PATH`, where `$ROOT_PATH` is an **absolute path** to the subject. In the case of above directory, `$ROOT_PATH` is `$ROOT/data/subjects/$SUBJECT_NAME`.

## 2. Preprocess `generated` data (necessary to create avatars)
* Go to `tools` folder.
* Run `python run_generated.py --root_path $ROOT_PATH`, where `$ROOT_PATH` is an **absolute path** to the subject. In the case of above directory, `$ROOT_PATH` is `$ROOT/data/subjects/$SUBJECT_NAME`.

## 3. Preprocess `test` data (only for quantitative evaluations of NeuMan and X-Humans datasets)
* Go to `tools/NeuMan` or `tools/XHumans` folder and run `python prepare_captured_test_imgs.py --root_path $DB_PATH --save_path $ROOT/data/subjects/$SUBJECT_NAME` where `$DB_PATH` represents the original dataset path.
* Training/testing split files of `NeuMan` dataset is available in [here](https://drive.google.com/drive/folders/1QJztYKjI9tC90U6mELwt-RF1V09LBqjW?usp=sharing).
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
* [Gyeongsik](https://drive.google.com/file/d/1Pbt6BL-trSFGEBGL6xGCa3q47TSlky5T/view?usp=sharing).
* [Loose clothing 1](https://drive.google.com/file/d/1gRES16bj4-qi6aBZvq_PX4A4oGxeYjxe/view?usp=sharing).
* [Loose_clothing_2](https://drive.google.com/file/d/1edZh7LvoLAHjtlWagJCE-cxb77irh78d/view?usp=sharing).

## Reference
```
@inproceedings{sim2025persona,
  title={{PERSONA}: Personalized Whole-Body {3D} Avatar with Pose-Driven Deformations from a Single Image},
  author = {Sim, Geonhee and Moon, Gyeongsik},  
  booktitle={ICCV},
  year={2025}
}
```
