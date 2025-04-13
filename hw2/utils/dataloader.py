import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

def collate_fn(batch):
    return tuple(zip(*batch))


class DigitDetectionDataset(Dataset):
    def __init__(self, data_root_path, mode="train", transforms=None):
        self.image_dir = os.path.join(data_root_path, mode)
        self.transforms = transforms
        self.mode = mode

        if self.mode in ["train", "val", "valid"]:
            annotation_file_name = f"{self.mode}.json"
            annotation_path = os.path.join(data_root_path, annotation_file_name)
            with open(annotation_path) as f:
                data = json.load(f)
            self.images = data['images']
            self.annotations = data['annotations']
            self.image_id_to_annots = {}
            for ann in self.annotations:
                self.image_id_to_annots.setdefault(ann['image_id'], []).append(ann)
        else:
            self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.*")))

    def __getitem__(self, idx):
        if self.mode in ["train", "val", "valid"]:
            image_info = self.images[idx]
            img_path = os.path.join(self.image_dir, image_info['file_name'])
            img = Image.open(img_path).convert("RGB")

            annots = self.image_id_to_annots.get(image_info['id'], [])
            boxes = []
            labels = []

            for a in annots:
                boxes.append(a['bbox'])
                labels.append(a['category_id'])

            if self.transforms:
                img_tensor = self.transforms(img)

            boxes_pixel = []
            for x, y, w, h in boxes:
                x1, y1, x2, y2 = x, y, x + w, y + h
                boxes_pixel.append([x1, y1, x2, y2])

            target = {
                'boxes': torch.tensor(boxes_pixel, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([image_info['id']])
            }

            return img_tensor, target

        else:

            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert("RGB")

            if self.transforms:
                img_tensor = self.transforms(img)

            return img_tensor, {"image_id": os.path.basename(img_path).split('.')[0]}



    def __len__(self):
        return len(self.images) if self.mode in ["train", "val", "valid"] else len(self.image_paths)
