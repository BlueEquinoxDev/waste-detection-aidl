from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch
import random
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from custom_datasets.taco_dataset_mask_r_cnn_update import TacoRCNNDataset
from torchvision.transforms.v2 import functional as F
from PIL import Image, ImageDraw
from collections import OrderedDict


CHECKPOINT_PATH = "app/checkpoint/checkpoint_epoch_30_2025_2_3_18_7.pt"
pic_idx = 5

data_transforms_validation = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor()
    ])
test_dataset=TacoRCNNDataset(annotations_file="data/test_annotations.json", img_dir="data/images", transforms=data_transforms_validation)
idx2class = test_dataset.idx_to_class
num_classes = len(idx2class)

model = maskrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torch.nn.Sequential(OrderedDict([
                ("conv5_mask", torch.nn.ConvTranspose2d(in_features_mask, hidden_layer, kernel_size=2, stride=2)),
                ("relu", torch.nn.ReLU()),
                ("mask_fcn_logits", torch.nn.Conv2d(hidden_layer, num_classes, kernel_size=1, stride=1)),
            ]))

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)

model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(list(model.roi_heads.box_predictor.parameters()) + list(model.roi_heads.mask_predictor.parameters()), lr=1e-3, weight_decay=0.0001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print([k for k in checkpoint.keys()])
#train_loss = checkpoint["train_loss"]
#val_loss = checkpoint["val_losses_avg"]

# Test the result model with one image ========================================================
print("TESTING TIME!!!")
# Get 1 image from dataset
im, target = test_dataset[pic_idx]
#im, target = test_dataset[0]
print(im.shape)
model.eval().to("cpu")
im = im.to("cpu")
print("Stating Inference")
detections = model([im])
print(detections)

iou_threshold = 0.2
scores_threshold = 0.6
detections = detections[0]
keep_idx = torchvision.ops.nms(detections["boxes"], detections["scores"], iou_threshold)
boxes = [b for i, b in enumerate(detections["boxes"]) if i in keep_idx]
scores = [s for i, s in enumerate(detections["scores"]) if i in keep_idx]
labels = [l for i, l in enumerate(detections["labels"]) if i in keep_idx]
masks = [m for i, m in enumerate(detections["masks"]) if i in keep_idx]

im = torchvision.transforms.functional.to_pil_image(im)
im_copy = im.copy()
for box, score, label, mask in zip(boxes, scores, labels, masks):
    if score > scores_threshold:
        mask = (mask[0].cpu().detach().numpy()*128).astype(np.int8)
        mask_im = Image.fromarray(mask, mode="L")
        full_color = Image.new("RGB", im_copy.size, (0, 255, 0))
        im_copy = Image.composite(full_color, im_copy, mask_im)

# Get COCO labels
named_labels = [v for k, v in idx2class.items()]
"""
with open("labels.txt", "r") as f:
    for line in f.readlines():
        coco_labels.append(line.replace("\n", ""))
"""
draw = ImageDraw.Draw(im_copy)
for box, score, label, mask in zip(boxes, scores, labels, masks):
    if score > scores_threshold:
        coords = box.cpu().tolist()
        draw.rectangle(coords, width=1, outline=(0, 255, 0))
        text = f"{named_labels[label]} {score*100:.2f}%"
        draw.text([coords[0], coords[1]-20], text, fill=(0, 255, 0), font_size=20)
im_copy.show()
