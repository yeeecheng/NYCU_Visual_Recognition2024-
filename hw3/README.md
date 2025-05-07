# NYCU Computer Vision 2025 Spring HW3

**StudentID**: 313553014 <br>
**Name**: 廖怡誠

## Introduction

This assignment tackles the challenging task of instance segmentation in medical imagery, which requires detecting and segmenting individual cell instances across four categories. Unlike object detection or semantic segmentation alone, this task demands per-instance, pixel-level labeling—even for overlapping or same-class objects. The dataset includes 209 training/validation and 101 test images, each annotated with binary masks indicating cell instances.

To address this under strict constraints (vision-only models, <200M parameters, no external data), I build upon Mask R-CNN and improve it through:

- Tiling large images to fit GPU memory and ignore empty regions,

- Albumentations-based augmentation for improved robustness,

- Refined training strategies (warm-up, learning rate tuning, weighted losses),

- COCO-style output formatting for evaluation and CodaBench submission.

These methods ensure a strong balance between accuracy, efficiency, and generalization, helping the model perform well under competitive settings.

The example of images in datasets after data augmentation:

<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="./src/demo1.png" alt="Picture 1" width="100%"/><br/>
      <strong>Pic 1.</strong>
    </td>
    <td align="center" width="34%">
      <img src="./src/demo2.png" alt="Picture 2" width="100%"/><br/>
      <strong>Pic 2.</strong>
    </td>
    <td align="center" width="33%">
      <img src="./src/demo3.png" alt="Picture 3" width="100%"/><br/>
      <strong>Pic 3.</strong>
    </td>
  </tr>
</table>


## How to install

```
git clone https://github.com/yeeecheng/NYCU_Visual_Recognition2024-.git
cd NYCU_Visual_Recognition2024-/hw3
conda create -n hw3 python=3.9 -y
conda activate hw3
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
* train
```
bash ./script/train.sh
```
You can modify the augments usage:

python main.py [-h] [--lr LR] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--num-classes NUM_CLASSES]
               [--data-path DATA_PATH] [--output-dir OUTPUT_DIR] [--backend BACKEND]
* test
```
bash ./script/inference.sh
```
You can modify the augments usage:

python inference.py [-h] --ckpt CKPT [--test-dir TEST_DIR] [--id-map ID_MAP] [--output OUTPUT]
                    [--score-thresh SCORE_THRESH]


## Performence snapshot

After extensive ablation and training analysis, the final model configuration uses a ResNet-50 backbone, default anchor sizes, and no learning rate scheduler. Removing StepLR improved convergence by preventing premature learning rate decay, allowing better optimization and lower final loss.

Anchor size experiments showed that smaller anchors underperformed due to the large cell structures in high-resolution images—making the default anchors more effective for region proposals.

Although a ResNet-101 backbone showed early promise (val loss down to 0.35), it required 512×512 tiling due to memory limits and couldn't be fully trained in time. Based on its early trend, it's likely to outperform ResNet-50 with sufficient training and overlapping tiles.

- Final score: 0.39 (private leaderboard) with ResNet-50
- Estimated potential: Higher with full ResNet-101 training

<div align="center">
  <img src="./src/per1.png" alt="Performance 1" width="60%"><br/>
  <img src="./src/per2.png" alt="Performance 2" width="40%"><br/>
  <img src="./src/per3.png" alt="Performance 3" width="60%">
</div>
