from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from collections import OrderedDict
import torch
import torch.nn as nn

class WasteMaskRCNN(nn.Module):
    def __init__(self, num_classes:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")


        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = torch.nn.Sequential(OrderedDict([
                        ("conv5_mask", torch.nn.ConvTranspose2d(in_features_mask, hidden_layer, kernel_size=2, stride=2)),
                        ("dropout", torch.nn.Dropout(0.5)),
                        ("relu", torch.nn.ReLU()),
                        ("mask_fcn_logits", torch.nn.Conv2d(hidden_layer, num_classes, kernel_size=1, stride=1)),
                    ]))

