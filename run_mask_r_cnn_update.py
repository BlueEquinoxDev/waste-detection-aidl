from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.roi_heads import fastrcnn_loss
import torchvision
import torch
from torch import Tensor
import random
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from custom_datasets.taco_dataset_mask_r_cnn_update import TacoRCNNDataset
from torchvision.transforms.v2 import functional as F
from PIL import Image, ImageDraw
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional
import os
import datetime
#from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

sed=23
torch.manual_seed(sed)
random.seed(sed)
np.random.seed(sed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# else torch.device("mps") if torch.backends.mps.is_available() 
#print(f"Device: {device}")

data_transforms_train = transforms.Compose([            
    #transforms.RandomResizedCrop(size=(500, 500), antialias=True, interpolation=transforms.InterpolationMode.NEAREST),
    #transforms.Resize(size=(500, 500), antialias=True, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToDtype(torch.float32, scale=True),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ToPureTensor()
    ])

data_transforms_validation = transforms.Compose([      
    #transforms.Resize(size=(500, 500), antialias=True, interpolation=transforms.InterpolationMode.NEAREST),      
    transforms.ToDtype(torch.float32, scale=True),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.ToPureTensor()
    ])

train_dataset=TacoRCNNDataset(annotations_file="data/train_annotations.json",
                                       img_dir="data/images",
                                       transforms=data_transforms_train)

idx2class = train_dataset.idx_to_class
#train_dataset = torch.utils.data.Subset(train_dataset, range(0, 1, 1))


validation_dataset=TacoRCNNDataset(annotations_file="data/validation_annotations.json",
                                            img_dir="data/images",
                                            transforms=data_transforms_validation)
#validation_dataset = torch.utils.data.Subset(validation_dataset, range(0, 1, 1))


test_dataset=TacoRCNNDataset(annotations_file="data/test_annotations.json",
                                            img_dir="data/images",
                                            transforms=data_transforms_validation)
#test_dataset = torch.utils.data.Subset(test_dataset, range(0, 1, 1))

def collate_fn(batch):
    images = []
    targets = []
    for img, tg in batch:
        images.append(img)
        targets.append(tg)
    return images, targets


train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=0, collate_fn=collate_fn)
val_data_loader = DataLoader(validation_dataset, shuffle=False, batch_size=2, num_workers=0, collate_fn=collate_fn)
test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=2, num_workers=0, collate_fn=collate_fn)

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


optimizer = torch.optim.Adam(list(model.roi_heads.box_predictor.parameters()) + list(model.roi_heads.mask_predictor.parameters()), lr=1e-3, weight_decay=0.0001)

def train_one_epoch():
    """ Train one epoch """
    model.train()
    model.to(device)
    losses_avg=0
    # Train one epoch
    for i, (images, targets) in enumerate(train_data_loader):
        model.zero_grad()
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

        losses_avg += loss.item()

        if i%2 == 0:
            loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
            print(f"Training [{i}/{len(train_data_loader)}] loss: {loss_dict_printable}")
    return losses_avg/len(train_data_loader) # Remove the [:2] to train the entire dataset

def eval_forward(model, images, targets, device):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()
    model.to(device)
    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )
    
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True

    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections

def evaluate_loss(model, data_loader, device):
    val_loss = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict, detections = eval_forward(model, images, targets, device)
         
            loss = sum(loss for loss in loss_dict.values())

            val_loss += loss

            if i%2 == 0:
                loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
                print(f"Validation [{i}/{len(data_loader)}] loss: {loss_dict_printable}")
          
    validation_loss = val_loss/ len(data_loader)    
    return validation_loss


"""def validation_one_epoch():      
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_data_loader):
            print(f"images [{images[0].shape}]")       
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]         
            
            loss_dict = model(images, targets)
            #print(f"predictions: {predictions}")
            print(f"loss_dict: {loss_dict}")
            loss = sum(loss for loss in loss_dict.values())
                
            loss_dict_printable = {k: f"Validation {v.item():.2f}" for k, v in loss_dict.items()}
            print(f"[{i}/{len(val_data_loader)}] loss: {loss_dict_printable}")"""


NUM_EPOCH=50
all_loss=[]
for epoch in range(1,NUM_EPOCH+1):
    train_losses_avg=train_one_epoch()
    val_losses_avg = evaluate_loss(model, val_data_loader, device)
    print(f"Epoch[{epoch}/{NUM_EPOCH}]: Train avg. loss:{train_losses_avg:.3f} Val avg. loss: {val_losses_avg:.3f}")
    all_loss.append((train_losses_avg, val_losses_avg.item()))

    checkpoint = {
            "model_state_dict":  model.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses_avg,
            "val_loss": val_losses_avg,
            "idx2classes": idx2class
    }  

    if not os.path.exists(f"{os.getcwd()}/app/checkpoint/"):
        os.makedirs(f"{os.getcwd()}/app/checkpoint/")

    # using now() to get current time
    current_time = datetime.datetime.now()
    print("The attributes of now() are :")
    filename = f"{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"

    torch.save(checkpoint, f"{os.getcwd()}/app/checkpoint/checkpoint_epoch_{str(epoch).zfill(3)}_{filename}.pt")
print(all_loss)

# Test the result model with one image ========================================================
print("TESTING TIME!!!")
# Get 1 image from dataset
im, target = train_dataset[0]
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
