import os
import glob
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, mode= "train", transform= None):
        
        self.root_dir = os.path.join(root_dir, mode)
        self.mode = mode
        self.transform = transform

        self.image_paths = []
        self.labels = []

        if self.mode in ["train", "val"]:
            self.classes = sorted(os.listdir(self.root_dir))

            for class_id in self.classes:
                class_path = os.path.join(self.root_dir, class_id)
                for img_path in glob.glob(os.path.join(class_path, "*.*")):
                    # print(img_path, class_id)
                    self.image_paths.append(img_path)
                    self.labels.append(class_id)
        else:
            for img_path in glob.glob(os.path.join(self.root_dir, "*.*")):
                self.image_paths.append(img_path)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = int(self.labels[idx]) if self.mode in ["train", "val"] else None
        
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)


