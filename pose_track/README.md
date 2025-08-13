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
* If your original data do not have temporal space (set of seprated images), follow the first directoty. Otherwise, follow the second directory.
* If you want only partial frames from videos, you can optionally save `valid_frame_list.txt` with your desired frame indices with newline for each frame index. For example, if you want 0th, 1st, 2nd, and 100th frames, `valid_frame_list.txt` should include `0\n1\n2\n100\n`.

## Start
* (Optional) If you want to specify bbox of the person in the first frame, prepare `$ROOT/data/subjects/$SUBJECT_NAME/bboxes/$FRAME_IDX.json` where `$FRAME_IDX` is the first frame index (e.g., 0). The json file should include [xmin, ymin, width, height].
* Run `python run.py --root_path $PATH --data_format $DATA_FORMAT`.
* `$PATH` is an **absolute path**. An example of `$PATH` of above directory is `$ROOT/data/subjects/$SUBJECT_NAME`.
* `$DATA_FORMAT` is one of `image` or `video`. If your original data do not have temporal space (set of seprated images), follow the first directoty and set `$DATA_FORMAT` to `image`. Otherwise, follow the second directory and set `$DATA_FORMAT` to `video`.
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

