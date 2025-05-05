import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()

        # Load the base model with pretrained weights
        self.backbone = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

        # Replace the box predictor head
        in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
        self.backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor head
        in_features_mask = self.backbone.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.backbone.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)

    def forward(self, images, targets=None):
        """
        Args:
            images (List[Tensor]): images to be processed
            targets (List[Dict], optional): ground-truth boxes and masks

        Returns:
            result (List[Dict] or Dict): the output from the model
        """
        return self.backbone(images, targets)
