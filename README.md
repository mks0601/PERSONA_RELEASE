# PERSONA (ICCV 2025)

## [Project Page](https://mks0601.github.io/PERSONA) | [Paper](https://mks0601.github.io/PERSONA) | [Video](https://www.youtube.com/watch?v=goQGvU_AwiU) 

* PERSONA is a personalized whole-body animatable avatar creation system that supports pose-driven deformations (*i.e.*, non-rigid deformations of clothes) from a single image.

<p align="middle">
<img src="assets/comparison_3d.gif" width="960" height="400">
</p>
<p align="center">
Compared to previous 3D-based methods, our PERSONA better represents non-rigid deformations of clothes.
For more high-resolution demo videos, please visit our <A href="https://mks0601.github.io/PERSONA">website</A>.
</p>

<p align="middle">
<img src="assets/comparison_gen.gif" width="960" height="400">
</p>
<p align="center">
Compared to previous diffusion-based methods, our PERSONA better preserves identity of the person.
For more high-resolution demo videos, please visit our <A href="https://mks0601.github.io/PERSONA">website</A>.
</p>


## Install
* To install a conda environment and necessary packages, run below.
```
conda env create -f environment.yml
conda activate persona
pip install -r requirements.txt
```

* Also, to download and install third-party modules, first, please get granted to access [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) and [sapiens pose](https://huggingface.co/noahcao/sapiens-pose-coco).
* Then, login HuggingFace on your machine by running `huggingface-cli login`.
* Finally, run below.
```
bash install.sh
```

## Creating and animating avatars from a single image
1. To create an avatar, you first need to fit SMPL-X to an image and generate training videos. Go to [./preprocess](./preprocess/).
2. Then, go to [./avatar](./avatar) to create and animate the avatar.

## Tracking 3D whole-body poses from a video
* To animate your avatar, you need to track 3D whole-body poses from a video. Go to [./pose_track](./pose_track).

## Reference
```
@inproceedings{sim2025persona,
  title={{PERSONA}: Personalized Whole-Body {3D} Avatar with Pose-Driven Deformations from a Single Image},
  author = {Sim, Geonhee and Moon, Gyeongsik},  
  booktitle={ICCV},
  year={2025}
}
```

