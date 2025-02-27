#see https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py#L67 for metrics
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


h_params ={
    "batch_size": 2,
    "num_workers": 0,
}

seed=23
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logdir = os.path.join("logs", f"segmentation-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

# TODO: Initialize Tensorboard Writer with the previous folder 'logdir'
writer=SummaryWriter(log_dir=logdir)

# Save checkpoint function
def save_model(model, epoch):
    checkpoint = {
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "idx2classes": idx2class
    }  

    save_path = os.path.join("results","mask_rcnn")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # using now() to get current time
    current_time = datetime.now()
    filename = f"{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"

    torch.save(checkpoint, os.path.join(save_path, f"checkpoint_epoch_{epoch}_{filename}.pt"))

# Create the datasets for train and validate
data_transforms_train = transforms.Compose([            
    transforms.Resize((800,800)),    
    transforms.ColorJitter(brightness=0.5, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation= transforms.InterpolationMode.BILINEAR, fill=0),
    transforms.RandomRotation(90, interpolation= transforms.InterpolationMode.NEAREST , expand=True, center=None, fill=0),
    transforms.RandomRotation(180, interpolation= transforms.InterpolationMode.NEAREST, expand=True, center=None, fill=0),
    transforms.RandomRotation(270, interpolation= transforms.InterpolationMode.NEAREST, expand=True, center=None, fill=0),
    transforms.RandomVerticalFlip(p=0.5),
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

# Generate Dataloaders for train & validation
train_loader=DataLoader(train_taco_dataset,
                        shuffle=True,
                        batch_size=h_params["batch_size"],
                        num_workers=h_params["num_workers"],
                        collate_fn=collate_fn)

valiation_loader=DataLoader(validation_taco_dataset,
                            shuffle=False,
                            batch_size=h_params["batch_size"], 
                            num_workers=h_params["num_workers"],
                            collate_fn=collate_fn)
 
# Import model and assign the nr of classes
model=WasteMaskRCNN(num_classes=len(train_taco_dataset.idx2class))
model.to(device)

# Create the optimizer
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
optimizer=torch.optim.AdamW(model.parameters(),
                            lr=1e-4,
                            weight_decay=1e-2)


def train_one_epoch():
    """
    Function to train 1 epoch of Mask R-CNN
    returns:
        -  avg loss in training
    """
    model.train()
    losses_avg=0
    len_dataset=len(train_loader)  
    for  batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data    
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}      
        print(f"[{batch}/{len_dataset}] total loss: {losses.item():.2f} losses: {loss_dict_printable}")     
        losses_avg+= losses.item()
        
    return losses_avg/len_dataset
        
        

def validation_one_epoch():
    """
    Function to validate 1 epoch of Mask R-CNN
    returns:
        -  avg loss in validation
    """
    losses_avg=0
    len_dataset=len(valiation_loader)  
    for  batch, data in enumerate(valiation_loader):
        images,targets=data            
        images=list(image.to(device) for image in images)   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
    
            loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}      
            print(f"[{batch}/{len_dataset}] total loss: {losses.item():.2f} losses: {loss_dict_printable}")     
            losses_avg+=losses.item()    

    return losses_avg/len_dataset
    

### START TRAINING
print("STARING TRAINING")
NUM_EPOCH=25
train_loss=[]
validation_loss=[]
for epoch in range(1,NUM_EPOCH+1):
    losses_avg_train=train_one_epoch()
    losses_avg_validation=validation_one_epoch()
    print(f"TRAINING epoch[{epoch}/{NUM_EPOCH}]: avg. loss: {losses_avg_train:.3f}")
    print(f"VALIDATION epoch[{epoch}/{NUM_EPOCH}]: avg. loss:{ losses_avg_validation:.3f}")  
    train_loss.append(losses_avg_train)
    validation_loss.append(losses_avg_validation)
    save_model(model, epoch=epoch)    
    writer.add_scalar('Segmentation/train_loss', losses_avg_train, epoch)
    writer.add_scalar('Segmentation/val_loss', losses_avg_validation, epoch)

print("Final train loss:\n")
print(validation_loss)

print("Final validation loss:\n")
print(validation_loss)

### START EVALUATION
print("STARING EVALUATION")
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


for images, targets in enumerate(test_loader):
    detections, metrics = model.evaluate(images=images, targets=targets)


print("Final test accuracy:\n")
print(f"Metrics: {metrics}")





