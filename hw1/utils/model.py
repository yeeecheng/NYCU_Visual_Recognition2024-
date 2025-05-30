import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")

class Resnet152(nn.Module):
    def __init__(self, num_classes=100, mode="train"):
        super(Resnet152, self).__init__()

        if mode == "train":
            self.resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
           self.resnet = resnet152(weights=None)
    
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)