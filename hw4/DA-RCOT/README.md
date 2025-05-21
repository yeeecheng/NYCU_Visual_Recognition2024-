# (TPAMI 25) Degradation-Aware Residual-Conditioned Optimal Transport for All-in-One Image Restoration
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2411.01656)

<hr />

> **Abstract:** *Unified, or more formally, all-in-one image restoration has emerged as a practical and promising low-level vision task for real-world applications. In this context, the key issue lies in how to deal with different types of degraded images simultaneously. Existing methods fit joint regression models over multi-domain degraded-clean image pairs of different degradations. However, due to the severe ill-posedness of inverting heterogeneous degradations, they often struggle with thoroughly perceiving the degradation semantics and rely on paired data for supervised training, yielding suboptimal restoration maps with structurally compromised results and lacking practicality for real-world or unpaired data. To break the barriers, we present a Degradation-Aware Residual-Conditioned Optimal Transport (DA-RCOT) approach that models (all-in-one) image restoration as an optimal transport (OT) problem for unpaired and paired settings, introducing the transport residual as a degradation-specific cue for both the transport cost and the transport map. Specifically, we formalize image restoration with a residual-guided OT objective by exploiting the degradation-specific patterns of the Fourier residual in the transport cost. More crucially, we design the transport map for restoration as a two-pass DA-RCOT map, in which the transport residual is computed in the first pass and then encoded as multi-scale residual embeddings to condition the second-pass restoration. This conditioning process injects intrinsic degradation knowledge (e.g., degradation type and level) and structural information from the multi-scale residual embeddings into the OT map, which thereby can dynamically adjust its behaviors for all-in-one restoration. Extensive experiments across five degradations demonstrate the favorable performance of DA-RCOT as compared to state-of-the-art methods, in terms of distortion measures, perceptual quality, and image structure preservation. Notably, DA-RCOT delivers superior adaptability to real-world scenarios even with multiple degradations and shows distinctive robustness to both degradation levels and the number of degradations.*
<hr />

##  Setup

###  Dependencies Installation

1. Clone our repository
```
git clone https://github.com/xl-tang3/DA-RCOT.git
cd DA-RCOT
```

2. Create conda environment
```
conda create -n DARCOT python=3.8
conda activate DARCOT
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
pip install -r requirements.txt
```

###  Dataset Download and Preperation

All the datasets used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing), [Kodak24]([https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u](https://www.kaggle.com/datasets/drxinchengzhu/kodak24/data)), [BSD68](https://github.com/cszn/DnCNN/tree/master/testsets)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (SOTS)

Deblurring: [GoPro](https://seungjunnah.github.io/Datasets/gopro.html)

Low-light enhancement: [LOLv1](https://github.com/caiyuanhao1998/Retinexformer)

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise, Derain, Dehaze or any single degradation.
After placing the training data the directory structure would be as follows:
```
└───Train
    ├───Dehaze
    │   ├───original
    │   └───synthetic
    ├───Denoise
    └───Derain
        ├───gt
        └───rainy
    └───Deblur
        ├───blur
        ├───sharp
    └───low_light
        ├───high
        ├───low
    └───single
    │   ├───degraded
    │   └───target
```

The testing data should be placed in the ```test``` directory wherein each task has a separate directory. The test directory after setup:

```
└───Test
    ├───dehaze
    │   ├───input
    │   └───target
    ├───denoise
    │   ├───bsd68
    │   └───kodak24
    ├───deblur
    │   ├───input
    │   └───target
    ├───lowlight
    │   ├───low
    │   └───high
    └───derain
    │   └───Rain100L
    │        ├───input
    │        └───target
```
### Training
The backbone **MRCNet** employs Restormer as the base architecture, integrated with our two-pass multi-scale residual design. condition mechanism.

#### 3 Degradation example:

```
python trainer.py --batchSize=2 --nEpochs=50 --pairnum=10000000 --Sigma=10000 --sigma=1 --de_type derain dehaze denoise_15 denoise_25 denoise_50 --patch_size=128  --type all --gpus=0 --backbone=MRCNet --step=15 --resume=none
```

#### 5 Degradation example:

```
python trainer.py --batchSize=2 --nEpochs=55 --pairnum=10000000 --Sigma=10000 --sigma=1 --de_type derain dehaze denoise_15 denoise_25 denoise_50 deblur lowlight --patch_size=128 --type all --gpus=0 --backbone=MRCNet --step=15 --resume=none
```

####




## Citation
```
@article{tang2025degradation,
  author={Tang, Xiaole and Gu, Xiang and He, Xiaoyi and Hu, Xin and Sun, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Degradation-Aware Residual-Conditioned Optimal Transport for Unified Image Restoration},
  year={2025},
  volume={},
  number={},
  pages={1-16}}

```

Contact me at Sherlock315@163.com if there is any problem.


