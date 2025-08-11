# PERSONA (ICCV 2025)

## [Project Page](https://mks0601.github.io/PERSONA) | [Paper](https://mks0601.github.io/PERSONA) | [Video](https://mks0601.github.io/PERSONA) 

* PERSONA is a personalized whole-body animatable avatar creation system that supports pose-driven deformations (*i.e.*, non-rigid deformations of clothes) from a single image.

<p align="middle">
<img src="assets/teaser_compressed.gif" width="960" height="400">
</p>
<p align="center">
Yes, it's me, Gyeongsik in the video :), taken in front of my apartment with my mobile phone.
For more high-resolution demo videos, please visit our <A href="https://mks0601.github.io/PERSONA">website</A>.
</p>

## Install
* To install a conda environment and necessary packages, run below.
```
conda env create -f environment.yml
conda activate persona
pip install -r requirements.txt
```
* Also, to download and install third-party modules, run below.
```
sh install.sh
```

## Creating and animating avatars from a single image
1. To create an avatar, you first need to fit SMPL-X to an image and generate training videos. Go to [here](./fitting/).
2. Then, go to [here](./avatar) to create and animate the avatar.

## Tracking 3D whole-body poses from a video
* To animate your avatar, you need to track 3D whole-body poses from a video. Go to [here](./pose_track/).

## Reference
```
@inproceedings{sim2025persona,
  title={{PERSONA}: Personalized Whole-Body {3D} Avatar with Pose-Driven Deformations from a Single Image},
  author = {Sim, Geonhee and Moon, Gyeongsik},  
  booktitle={ICCV},
  year={2025}
}
```

