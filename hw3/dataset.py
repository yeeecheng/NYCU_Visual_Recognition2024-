import os
import cv2
import glob
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class MedicalInstanceDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.samples = sorted(os.listdir(root_dir))
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_dir = os.path.join(self.root_dir, self.samples[idx])
        img_path = os.path.join(data_dir, 'image.tif')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_np = img  # HWC 格式

        mask_paths = sorted(glob.glob(os.path.join(data_dir, 'class*.tif')))
        masks, labels = [], []
        for mp in mask_paths:
            class_id = int(os.path.basename(mp).split('class')[1].split('.tif')[0])
            mask_img = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                continue
            instances = np.unique(mask_img)
            instances = instances[instances != 0]
            for inst_id in instances:
                bin_mask = (mask_img == inst_id).astype(np.uint8)
                masks.append(bin_mask)
                labels.append(class_id)

        if masks:
            masks = np.stack(masks)  # (N, H, W)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            h, w = image_np.shape[:2]
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)

        # --- Apply Albumentations transforms ---
        if self.transforms:
            transformed = self.transforms(image=image_np, masks=list(masks))
            image = transformed["image"]
            if transformed["masks"]:
                masks = torch.as_tensor(np.stack(transformed["masks"]), dtype=torch.uint8)
            else:
                masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
        else:
            image = TF.to_tensor(image_np.transpose(2, 0, 1))
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        # --- Generate boxes and filter invalid ones ---
        valid_boxes, valid_masks, valid_labels = [], [], []
        for i in range(len(masks)):
            pos = masks[i].nonzero()
            if pos.numel() == 0:
                continue
            xmin = torch.min(pos[:, 1])
            xmax = torch.max(pos[:, 1])
            ymin = torch.min(pos[:, 0])
            ymax = torch.max(pos[:, 0])
            if xmax > xmin and ymax > ymin:
                valid_boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
                valid_masks.append(masks[i])
                valid_labels.append(labels[i])

        if valid_boxes:
            boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
            masks = torch.stack(valid_masks)
            labels = torch.as_tensor(valid_labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
        }


        # import torchvision.utils as vutils
        # import matplotlib.pyplot as plt


        # debug_dir = "./debug_vis"
        # os.makedirs(debug_dir, exist_ok=True)

        # # 將 image 轉為 numpy 格式 [H, W, 3]
        # img_np = image.permute(1, 2, 0).numpy().copy()
        # img_np = (img_np * 255).astype(np.uint8)

        # # 畫上 bounding boxes
        # for box in boxes:
        #     x1, y1, x2, y2 = map(int, box)
        #     img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # # 顯示 mask 疊圖（optional: 最多顯示前 N 個）
        # for i, m in enumerate(masks[:3]):
        #     colored = np.zeros_like(img_np)
        #     colored[m.numpy() > 0] = (0, 0, 255)
        #     img_np = cv2.addWeighted(img_np, 1.0, colored, 0.5, 0)

        # # 儲存
        # cv2.imwrite(os.path.join(debug_dir, f"{idx:03d}.jpg"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))



        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))




