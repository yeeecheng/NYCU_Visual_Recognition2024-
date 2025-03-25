# NYCU Computer Vision 2025 Spring HW1

studentID: 313553014
Name: 廖怡誠

## Introduction

This project focuses on image classification of various plants and animals. Some images have complex or distracting backgrounds, making it hard for the model to learn the correct features.

For example:

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="text-align: center; width: 48%;">
    <img src="./src/img/pic1.png" alt="Picture 1" style="width: 100%;">
    <p><strong>Pic 1.</strong> Complex background case.</p>
  </div>
  <div style="text-align: center; width: 48%;">
    <img src="./src/img/pic2.png" alt="Picture 2" style="width: 100%;">
    <p><strong>Pic 2.</strong> Miss target object case.</p>
  </div>
</div>

To address this, I manually inspected the training images and found that the classification target is usually centered. Based on this, I cropped image borders to reduce background noise. I also applied extensive data augmentation—such as color jitter, random affine, erasing, and rotation—to improve generalization. Additionally, I tuned hyperparameters, optimizers, and learning rate schedules to further enhance performance.

## How to install

```
git clone https://github.com/yeeecheng/NYCU_Visual_Recognition2024-.git
cd hw1
conda create -n hw1 python=3.9 -y
conda activate hw1
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Performence snapshot
