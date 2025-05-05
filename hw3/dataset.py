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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(img.transpose(2, 0, 1))

        mask_paths = sorted(glob.glob(os.path.join(data_dir, 'class*.tif')))
        masks, labels = [], []
        for mp in mask_paths:
            class_id = int(os.path.basename(mp).split('class')[1].split('.tif')[0])
            mask_img = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            instances = np.unique(mask_img)
            instances = instances[instances != 0]
            for inst_id in instances:
                bin_mask = (mask_img == inst_id).astype(bool)
                masks.append(bin_mask)
                labels.append(class_id)

        if masks:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, image.size[1], image.size[0]), dtype=torch.uint8)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms:
            transformed = self.transforms(
                image=image.permute(1, 2, 0).numpy(),  # convert back to HWC numpy
                masks=[m.numpy() for m in masks]       # list of numpy 2D masks
            )
            image = TF.to_tensor(transformed["image"])
            masks = torch.as_tensor(np.stack(transformed["masks"]), dtype=torch.uint8)
        else:
            image = TF.to_tensor(image)


        boxes = []
        for m in masks:
            pos = m.nonzero()
            xmin = torch.min(pos[:, 1])
            xmax = torch.max(pos[:, 1])
            ymin = torch.min(pos[:, 0])
            ymax = torch.max(pos[:, 0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
        }

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))
