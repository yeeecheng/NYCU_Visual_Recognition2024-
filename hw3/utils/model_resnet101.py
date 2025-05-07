import torchvision
import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models import resnext101_32x8d
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN_ResNeXt101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load pretrained ResNeXt-101 backbone
        backbone = resnext101_32x8d(weights="DEFAULT")
        backbone = _resnet_fpn_extractor(backbone, trainable_layers=3)

        # Assemble into Mask R-CNN
        self.model = MaskRCNN(backbone, num_classes=num_classes)

        # Replace box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
