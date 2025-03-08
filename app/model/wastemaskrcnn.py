#https://pytorch.org/vision/0.19/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from utilities.compute_metrics import compute_dice, compute_iou


class WasteMaskRCNN(nn.Module):
    """
    Custom Mask RCNN for Waste
    """
    def __init__(self,
                 num_classes:int,
                 checkpoint_path: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
                
        self.num_classes = num_classes
        
        self.model = self.__get_model_instance_segmentation__(num_classes)  

        # If model checkpoint
        if checkpoint_path:
            self.checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            
        
    def forward(self,inputs,targets=None):
        if targets == None:
            self.model.eval()
            return self.model(inputs)
        return self.model(inputs,targets)
            
    def __get_model_instance_segmentation__(self, num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                                                      box_detections_per_img=512,
                                                                      trainable_backbone_layers=3)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        #for p in model.backbone.parameters():p.requires_grad=False
        return model
    
    def preprocessing(self, images, transformations=None):
        if transformations is None:
            images = [ImageOps.exif_transpose(img) for img in images]
            images = [tv_tensors.Image(img) for img in images]

            transformations = transforms.Compose([
                transforms.Resize((800,800)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.ToPureTensor()
            ])
        
        images = transformations(images)
        return images

    
    def evaluate(self, images:list, idx2class:dict, targets:list=None, transforms=None, preprocessing=False):
        if preprocessing:
            print(type(images))
            images = self.preprocessing(images=images, transformations=transforms)
        
        self.model.eval()
        with torch.no_grad():
            detections = self.model(images)
            #print("Print detections")
            #print(detections)
            if targets is not None:
                # Compute metrics
                metrics = None
                iou = compute_iou(detections, targets)
                dice = compute_dice(detections, targets)
                metrics = {
                    "iou": iou,
                    "dice": dice,
                }
                return detections, metrics
            else:
                iou_threshold = 0.2
                scores_threshold = 0.6
                processed_images = []
                processed_detections = []
    
                for detection_items, image in list(zip(detections, images)):
                    print("detections file in detections area")
                    
                    keep_idx = torchvision.ops.nms(detection_items["boxes"], detection_items["scores"], iou_threshold)
                    boxes = [b for i, b in enumerate(detection_items["boxes"]) if i in keep_idx]
                    scores = [s for i, s in enumerate(detection_items["scores"]) if i in keep_idx]
                    labels = [l for i, l in enumerate(detection_items["labels"]) if i in keep_idx]
                    masks = [m for i, m in enumerate(detection_items["masks"]) if i in keep_idx]

                    processed_boxes = [b.tolist() for i, b in enumerate(detection_items["boxes"]) if i in keep_idx]
                    processed_scores = [s.tolist() for i, s in enumerate(detection_items["scores"]) if i in keep_idx]
                    processed_labels = [l.tolist() for i, l in enumerate(detection_items["labels"]) if i in keep_idx]
                    processed_masks = [m.tolist() for i, m in enumerate(detection_items["masks"]) if i in keep_idx]

                    image = torchvision.transforms.functional.to_pil_image(image)
                    im_copy = image.copy()
                    #print(im_copy)
                    for box, score, label, mask in zip(boxes, scores, labels, masks):
                        if score > scores_threshold:
                            mask = (mask[0].cpu().detach().numpy()*128).astype(np.int8)
                            mask_im = Image.fromarray(mask, mode="L")
                            full_color = Image.new("RGB", im_copy.size, (0, 255, 0))
                            im_copy = Image.composite(full_color, im_copy, mask_im)

                    # Get COCO labels
                    named_labels = [v for k, v in idx2class.items()]

                    draw = ImageDraw.Draw(im_copy)
                    for box, score, label, mask in zip(boxes, scores, labels, masks):
                        if score > scores_threshold:
                            coords = box.cpu().tolist()
                            draw.rectangle(coords, width=1, outline=(0, 255, 0))
                            text = f"{named_labels[label]} {score*100:.2f}%"
                            draw.text([coords[0], coords[1]-20], text, fill=(0, 255, 0), font_size=20)
                    processed_images.append(im_copy)
                    processed_detections.append({
                        "boxes": processed_boxes,
                        "scores": processed_scores,
                        "labels": processed_labels,
                        "masks": processed_masks,
                    })
                
                return processed_detections, processed_images
            