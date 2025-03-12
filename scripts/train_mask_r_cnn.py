#see https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py#L67 for metrics
import torch.utils
import torch.utils.data
from custom_datasets.taco_dataset_mask_r_cnn import TacoDatasetMaskRCNN
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import torch
import numpy as np
import os
import random
from pycocotools.cocoeval import COCOeval
from model.wastemaskrcnn import WasteMaskRCNN
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import List
from torchvision.ops import nms
from pycocotools.mask import encode
import evaluate
from utilities.save_model import save_model
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.segmentation import MeanIoU


h_params ={
    "batch_size": 1,
    "num_workers": 0,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "model_name": "Mask_R-CNN",
    "dataset_name": "taco28",
    "backup_best_model": True,
    "load_checkpoint": True, #True,
    "checkpoint_path": "results/mask_rcnn/checkpoint_epoch_7_2025_3_12_9_10.pt"
}

# experiment_name contains model name, backbone_freeze, augmentation
experiment_name = "seg-"+h_params["model_name"]+"-"+h_params["dataset_name"]
print("Experiment name: ", experiment_name)

logdir = os.path.join("logs", f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
results_dir = os.path.join("results", f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

seed=23
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize Tensorboard Writer with the previous folder 'logdir'
writer=SummaryWriter(log_dir=logdir)

# Create the datasets for train and validate
data_transforms_train = transforms.Compose([            
    transforms.Resize((800,800)),    
    transforms.ColorJitter(brightness=0.5, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=0.05),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation= transforms.InterpolationMode.BILINEAR, fill=0),
    transforms.RandomRotation(90, interpolation= transforms.InterpolationMode.NEAREST , expand=True, center=None, fill=0),
    transforms.RandomRotation(180, interpolation= transforms.InterpolationMode.NEAREST, expand=True, center=None, fill=0),
    transforms.RandomRotation(270, interpolation= transforms.InterpolationMode.NEAREST, expand=True, center=None, fill=0),
    #transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    ])

data_transforms_validation = transforms.Compose([
    transforms.Resize((800,800)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor()
    ])

# Create a collate function to use in the dataloader
def collate_fn(batch):
    return tuple(zip(*batch))

# Generate datasets for train & validation
train_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/train_annotations.json",
                                       img_dir="data/images",
                                       transforms=data_transforms_train)

validation_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/validation_annotations.json",
                                            img_dir="data/images",
                                            transforms=data_transforms_validation)

# Creating a variable to store the dict of index to labels
idx2class = train_taco_dataset.idx2class

"""
number_of_samples_by_category={cat:len(train_taco_dataset.coco_data.catToImgs[cat]) for cat in train_taco_dataset.idx2class.keys() if cat!=0 }
number_of_samples=sum({cat:len(train_taco_dataset.coco_data.catToImgs[cat]) for cat in train_taco_dataset.idx2class.keys() if cat!=0}.values())
category_weights={cat:number_of_samples/value for cat,value in number_of_samples_by_category.items()}
samples_weights=[]

#Computing samples weights for WeightedRandomSampler
for idx in range(train_taco_dataset.len_dataset):
    weights_cat_in_image=[]
    img_id=train_taco_dataset.index_to_imageId[idx]                        
    img_coco_data = train_taco_dataset.coco_data.loadImgs([img_id])[0]
    annotations = train_taco_dataset.coco_data.imgToAnns[img_id]
    for ann in annotations:
        weights_cat_in_image.append(category_weights[ann['category_id']])
    samples_weights.append(max(weights_cat_in_image))

sampler=torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=train_taco_dataset.len_dataset, replacement=True)
"""
sampler=None

# Generate Dataloaders for train & validation
train_loader=DataLoader(train_taco_dataset,
                        shuffle=True,
                        batch_size=h_params["batch_size"],
                        num_workers=h_params["num_workers"],
                        collate_fn=collate_fn,
                        #sampler=sampler
                        )

valiation_loader=DataLoader(validation_taco_dataset,
                            shuffle=False,
                            batch_size=h_params["batch_size"], 
                            num_workers=h_params["num_workers"],
                            collate_fn=collate_fn)
 
# Import model and assign the nr of classes
model=WasteMaskRCNN(num_classes=len(train_taco_dataset.idx2class))
model.to(device)

if h_params["load_checkpoint"]:
    checkpoint_path = h_params["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

# Create the optimizer
optimizer=torch.optim.AdamW(model.parameters(),
                            lr=h_params["learning_rate"],
                            weight_decay=h_params["weight_decay"])
# Load optimizer state
if h_params["load_checkpoint"]:
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    optimizer.load_state_dict(optimizer_state_dict)

# Define the scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

#metric = evaluate.load("mean_iou")
mAP_train = MeanAveragePrecision(iou_type="segm")
#mIoU_train = MeanIoU(num_classes=len(idx2class))

def train_one_epoch():
    """
    Function to train 1 epoch of Mask R-CNN
    returns:
        -  avg loss in training
    """
    model.train()
    losses_avg=0
    len_dataset=len(train_loader)  

    pbar = tqdm(train_loader, desc="Computing loss for train dataset", leave=False)
    for batch_idx, data in enumerate(pbar):
        model.train()
        optimizer.zero_grad()
        images, targets = data    
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        #loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}      
        #print(f"[{batch_idx}/{len_dataset}] total loss: {losses.item():.2f} losses: {loss_dict_printable}")     
        losses_avg += losses.item() 

        with torch.no_grad():
            predicts = model(images)
        
        #references = segmentation_maps(masks=[t["masks"] for t in targets])
        #pred_maps = segmentation_maps(masks=[p["masks"].squeeze(dim=1) for p in predicts])
        #metric.add_batch(references=references, predictions=pred_maps)

        # Format predictions
            for pred in predicts:
                pred['masks'] = (pred['masks'].squeeze(1) > 0.5).byte()  # Convert to binary mask
            
            # Format targets
            """
            formatted_targets = []
            for target in targets:
                formatted_target = {
                    'boxes': target['boxes'],
                    'labels': target['labels'],
                    'masks': target['masks'].byte(),  # Convert to binary mask
                    'image_id': target['image_id']
                }
                formatted_targets.append(formatted_target)
            """
        mAP_train.update(predicts, targets)
        

    mAP_result = mAP_train.compute()
    print(mAP_result)
    #metrics = metric.compute(num_labels=len(idx2class), ignore_index=255, reduce_labels=True)
    return losses_avg/len_dataset, mAP_result

"""
def segmentation_maps(masks):
    segmentations = []
    for mask in masks:
        semantic_seg = torch.zeros((mask.shape[1], mask.shape[2]), dtype=torch.int64, device=mask.device)
        for i, mask_cls in enumerate(mask):
            # Assign a unique instance ID (i+1) to the mask region
            #print(f"i: {i}, mask.shape: {mask.shape}")
            semantic_seg[mask_cls > 0] = i + 1
        segmentations.append(semantic_seg)
    return segmentations

def apply_non_maximun_suppresion(prediction,score_threshold=0.5,iou_threshold = 0.5):
    
    #Step 1: remove background
    valid_indices = torch.where(prediction['labels'] != 0)[0]
    prediction['scores'] =prediction['scores'][valid_indices]
    prediction['boxes']=prediction['boxes'][valid_indices]
    prediction['labels']=prediction['labels'][valid_indices]
    prediction['masks']=prediction['masks'][valid_indices]
    
    # Step 2: Apply score threshold
    valid_indices = torch.where(prediction['scores'] >= score_threshold)[0]
    prediction['scores'] =prediction['scores'][valid_indices]
    prediction['boxes']=prediction['boxes'][valid_indices]
    prediction['labels']=prediction['labels'][valid_indices]
    prediction['masks']=prediction['masks'][valid_indices]
    
    # Step 3: Non-Maximum Suppression 
    indices=nms(prediction['boxes'],prediction['scores'],iou_threshold)
    prediction['boxes']=prediction['boxes'][indices]
    prediction['labels']=prediction['labels'][indices]
    prediction['masks']=prediction['masks'][indices]
    return prediction


def mask_to_coco_format(threshold,mask):
    a=mask.cpu().squeeze().numpy()
    a=(a>=threshold).astype(np.uint8)
    a=np.asfortranarray(a)
    return encode(a)
        
def bbox_to_coco_format(bbox:torch.Tensor)->None:
    bbox=bbox.cpu().numpy()
    return [bbox[0],bbox[1],(bbox[2]-bbox[0]),(bbox[3]-bbox[1])]

def coco_result_format(images_id:List[int], predictions:List[torch.Tensor],threshold:float = 0.4)->np.ndarray:
    #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #https://cocodataset.org/#format-results
    #https://cocodataset.org/#detection-eval
    results = []
    for i in range(len(images_id)):
        image_id=images_id[i]
        prediction=apply_non_maximun_suppresion(predictions[i],threshold)
        for i in range(len(prediction['labels'])):
            results.append({
                    "image_id":image_id,
                    "category_id":prediction['labels'][i].item(),
                    "bbox":bbox_to_coco_format(prediction['boxes'][i]),                
                    "score":prediction['scores'][i].item(),
                    "segmentation":mask_to_coco_format(threshold,prediction['masks'][i])})    
    return results
""" 

mAP_val = MeanAveragePrecision(iou_type="segm")

def validation_one_epoch():
    """
    Function to validate 1 epoch of Mask R-CNN
    returns:
        -  avg loss in validation
    """
    losses_avg=0
    len_dataset=len(valiation_loader)

    # results = []

    pbar = tqdm(valiation_loader, desc="Computing loss for validation dataset", leave=False)
    for batch_idx, data in enumerate(pbar):
        model.train()
        images,targets=data            
        images=list(image.to(device) for image in images)   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            #loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}      
            #print(f"[{batch}/{len_dataset}] total loss: {losses.item():.2f} losses: {loss_dict_printable}")     
            losses_avg+=losses.item()
        
        with torch.no_grad():
            predicts=model(images)
            
            # Format predictions
            for pred in predicts:
                pred['masks'] = (pred['masks'].squeeze(1) > 0.5).byte()  # Convert to binary mask
            """
            # Format targets
            formatted_targets = []
            for target in targets:
                formatted_target = {
                    'boxes': target['boxes'],
                    'labels': target['labels'],
                    'masks': target['masks'].byte(),  # Convert to binary mask
                    'image_id': target['image_id']
                }
                formatted_targets.append(formatted_target)
            """
        mAP_val.update(predicts, targets)
        
        
    """

        images_id=[t['image_id'] for t in targets]                
        results.extend(coco_result_format(images_id,predicts))
        
        references = segmentation_maps(masks=[t["masks"] for t in targets])
        pred_maps = segmentation_maps(masks=[p["masks"].squeeze(dim=1) for p in predicts])
        metric.add_batch(references=references, predictions=pred_maps)
        break

    if not results:
        stats = list(range(0, 12))
    else:    
        coco_result=valiation_loader.dataset.coco_data.loadRes(results)
        coco_eval=COCOeval(valiation_loader.dataset.coco_data, coco_result)    
        coco_eval.params.useCats=0    
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
    
    metrics = metric.compute(num_labels=len(idx2class), ignore_index=255, reduce_labels=True)
    """
    mAP_result = mAP_val.compute()

    return losses_avg/len_dataset, mAP_result #stats, metrics
    

### START TRAINING
print("STARING TRAINING")
NUM_EPOCH = h_params["num_epochs"]
train_loss = []
validation_loss = []
for epoch in range(1, NUM_EPOCH+1):
    losses_avg_train, metrics_train=train_one_epoch()
    losses_avg_validation, metrics_val=validation_one_epoch() # stats
    #scheduler.step(losses_avg_validation)
    print(f"TRAINING epoch[{epoch}/{NUM_EPOCH}]: avg. loss: {losses_avg_train:.3f}")
    print(f"VALIDATION epoch[{epoch}/{NUM_EPOCH}]: avg. loss:{ losses_avg_validation:.3f}")  
    train_loss.append(losses_avg_train)
    validation_loss.append(losses_avg_validation)
    if h_params["backup_best_model"]:
        if losses_avg_validation < best_val_loss:
            best_val_loss = losses_avg_validation
            save_model(model, epoch, optimizer, idx2class, os.path.join(results_dir, 'best_mask_rcnn_model.pth'))
        # Update in training loop after validation
        scheduler.step(losses_avg_validation)
    
    writer.add_scalar('Segmentation/train_loss', losses_avg_train, epoch)
    writer.add_scalar('Segmentation/val_loss', losses_avg_validation, epoch)


    """
    writer.add_scalar('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', stats[0], epoch)
    writer.add_scalar('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]', stats[1], epoch)
    writer.add_scalar('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]', stats[2], epoch)
    writer.add_scalar('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', stats[3], epoch)
    writer.add_scalar('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', stats[4], epoch)
    writer.add_scalar('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', stats[5], epoch)
    writer.add_scalar('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]', stats[6], epoch)
    writer.add_scalar('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]', stats[7], epoch)
    writer.add_scalar('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', stats[8], epoch)
    writer.add_scalar('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', stats[9], epoch)
    writer.add_scalar('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', stats[10], epoch)
    writer.add_scalar('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', stats[11], epoch)
    """
    
    for key, value in metrics_train.items():
        if isinstance(value, np.ndarray):
            writer.add_scalars(f'Segmentation/train_{key}', {str(i):v for i, v in enumerate(value)}, epoch)
        else:
            if key == 'classes':
                pass
            else:
                writer.add_scalar(f'Segmentation/train_{key}', value, epoch)
    
    for key, value in metrics_val.items():
        if isinstance(value, np.ndarray):
            writer.add_scalars(f'Segmentation/val_{key}', {str(i):v for i, v in enumerate(value)}, epoch)
        else:
            if key == 'classes':
                pass
            else:
                writer.add_scalar(f'Segmentation/val_{key}', value, epoch)
    
print("Final train loss:\n")
print(validation_loss)

print("Final validation loss:\n")
print(validation_loss)

### START EVALUATION
print("STARING EVALUATION on TEST DATASET...")
test_dataset=WasteMaskRCNN(annotations_file="data/test_annotations.json", 
                           img_dir="data/images", 
                           transforms=data_transforms_validation)
idx2class = test_dataset.idx_to_class
num_classes = len(idx2class)

test_loader=DataLoader(test_dataset,
                       shuffle=False,
                       batch_size=h_params["batch_size"], 
                       num_workers=h_params["num_workers"],
                       collate_fn=collate_fn)

mAP_test = MeanAveragePrecision(iou_type="segm")

pbar = tqdm(test_loader, desc="Computing metrics for test dataset", leave=False)
for batch_idx, data in enumerate(pbar):
    model.eval()
    images, targets = data            
    images=list(image.to(device) for image in images)   
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        
    with torch.no_grad():
        predicts = model(images)
            
        # Format predictions
        for pred in predicts:
            pred['masks'] = (pred['masks'].squeeze(1) > 0.5).byte()  # Convert to binary mask
            
        mAP_test.update(predicts, targets)
mAP_test_results = mAP_test.compute()

print("Final test metrics:\n")
for metric_name, metric_value in mAP_test_results:
    print(f"{metric_name}: {metric_value}")

