import os
import cv2
import glob
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class MedicalInstanceDatasetTile(Dataset):
    def __init__(self, root_dir, transforms=None, tile_size=512, tile_stride=256):
        self.root_dir = root_dir
        self.transforms = transforms
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.samples = sorted(os.listdir(root_dir))


        self.tiles = []
        for img_idx, name in enumerate(self.samples):
            img_path = os.path.join(root_dir, name, 'image.tif')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            for y in range(0, h, tile_stride):
                for x in range(0, w, tile_stride):
                    self.tiles.append((img_idx, x, y))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_idx, x0, y0 = self.tiles[idx]
        data_dir = os.path.join(self.root_dir, self.samples[img_idx])
        img_path = os.path.join(data_dir, 'image.tif')


        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_np = img[y0:y0 + self.tile_size, x0:x0 + self.tile_size]
        h, w = image_np.shape[:2]


        mask_paths = sorted(glob.glob(os.path.join(data_dir, 'class*.tif')))
        masks, labels = [], []

        for mp in mask_paths:
            class_id = int(os.path.basename(mp).split('class')[1].split('.tif')[0])
            full_mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            if full_mask is None:
                continue
            tile_mask = full_mask[y0:y0 + self.tile_size, x0:x0 + self.tile_size]
            instances = np.unique(tile_mask)
            instances = instances[instances != 0]
            for inst_id in instances:
                bin_mask = (tile_mask == inst_id).astype(np.uint8)
                if bin_mask.sum() == 0:
                    continue
                masks.append(bin_mask)
                labels.append(class_id)

        if masks:
            masks = np.stack(masks)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)


        if self.transforms and len(masks) > 0:
            transformed = self.transforms(image=image_np, masks=list(masks))
            image = transformed["image"]
            if transformed["masks"]:
                masks = torch.as_tensor(np.stack(transformed["masks"]), dtype=torch.uint8)
            else:
                masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
        else:
            image = TF.to_tensor(image_np)  # no .transpose()
            masks = torch.as_tensor(masks, dtype=torch.uint8)

        # 產 bounding boxes
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
            # Dummy annotation
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
            masks = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.uint8)
            labels = torch.tensor([0], dtype=torch.int64)  # 假設 class 0 是 background or ignored

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_idx]),
        }


        # import torchvision.utils as vutils
        # import matplotlib.pyplot as plt


        # debug_dir = "./debug_vis_tile"
        # os.makedirs(debug_dir, exist_ok=True)

        # #
        # img_np = image.permute(1, 2, 0).numpy().copy()
        # img_np = (img_np * 255).astype(np.uint8)

        # #  bounding boxes
        # for box in boxes:
        #     x1, y1, x2, y2 = map(int, box)
        #     img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # #
        # for i, m in enumerate(masks[:3]):
        #     colored = np.zeros_like(img_np)
        #     colored[m.numpy() > 0] = (0, 0, 255)
        #     img_np = cv2.addWeighted(img_np, 1.0, colored, 0.5, 0)

        # # 儲存
        # cv2.imwrite(os.path.join(debug_dir, f"{idx:03d}.jpg"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        return image, target


def collate_fn_tile(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))