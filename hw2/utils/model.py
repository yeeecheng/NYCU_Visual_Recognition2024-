import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=11, mode="train"):
        super(FasterRCNN, self).__init__()
        if mode == "train":
            self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, trainable_backbone_layers=5)
        else:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None)

        # def custom_anchor_generator():
        #     anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        #     aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        #     return AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # self.model.rpn.anchor_generator = custom_anchor_generator()
 
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)