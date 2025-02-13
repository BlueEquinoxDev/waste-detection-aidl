#https://pytorch.org/vision/0.19/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn as nn
class WasteMaskRCNN(nn.Module):
    def __init__(self,num_classes:int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_classes=num_classes
        self.model=self.__get_model_instance_segmentation__(num_classes)       
        self.iou_types=["bbox","segm"] 
        
    def forward(self,inputs,targets=None):
        if targets==None:
            self.model.eval()
            return self.model(inputs)
        return self.model(inputs,targets)        
    def __get_model_instance_segmentation__(self,num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                                                      box_detections_per_img=512,
                                                                      trainable_backbone_layers=5)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
        hidden_layer = 1024
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        #for p in model.backbone.parameters():p.requires_grad=False
        return model