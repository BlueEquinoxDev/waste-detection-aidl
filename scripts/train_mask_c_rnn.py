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

sed=23
torch.manual_seed(sed)
random.seed(sed)
np.random.seed(sed)

#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

#device=torch.device("cpu")

logdir = os.path.join("runs", f"segmentation-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

# TODO: Initialize Tensorboard Writer with the previous folder 'logdir'
writer=SummaryWriter(log_dir=logdir)

def save_model(model):
    checkpoint = {
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict()
    }  


    if not os.path.exists(f"{os.getcwd()}/app/checkpoint/"):
        os.makedirs(f"{os.getcwd()}/app/checkpoint/")

    torch.save(checkpoint, f"{os.getcwd()}/app/checkpoint/checkpoint.pt")


data_transforms_train = transforms.Compose([            
    transforms.Resize((800,800)),
    transforms.RandomHorizontalFlip(0.5),    
    transforms.ColorJitter(brightness=0.5, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=0.05),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    #transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

data_transforms_validation = transforms.Compose([
    transforms.Resize((800,800)),    
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    #transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

train_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/train_annotations.json",
                                       img_dir="data",
                                       transforms=data_transforms_train)

validation_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/validation_annotations.json",
                                            img_dir="data",
                                            transforms=data_transforms_validation)


train_loader=DataLoader(train_taco_dataset,shuffle=True,batch_size=1,collate_fn=collate_fn)
valiation_loader=DataLoader(validation_taco_dataset,shuffle=True,batch_size=1,collate_fn=collate_fn)
 

model=WasteMaskRCNN(num_classes=28+1)
model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
print(f"parameters to optimize: {len(params)}")
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)


#optimizer=torch.optim.AdamW(model.parameters(),betas=(0.9,0.95),eps=1e-8,lr=0.001,weight_decay=0.005)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-2)

# and a learning rate scheduler
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

scaler = torch.cuda.amp.GradScaler()  if device.type == 'cuda' else None

def train_one_epoch():  
    losses_avg=0
    len_dataset=len(train_loader)  
    for  batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        images,targets=data    
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        #loss_dict = model(images,targets)
        
        #with torch.autocast(device_type="cuda"):
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        #lr_scheduler.step()
        
        loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}

        # reduce losses over all GPUs for logging purposes
        #loss_dict_reduced = reduce_dict(loss_dict)       
        #losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        #loss_value = losses_reduced.item()
        #loss_dict_reduced={key:loss_dict_reduced[key].item() for key in  loss_dict_reduced.keys()}
        

        
        
        #print(f"batch: {batch}, loss:{loss_value} losses: {loss_dict_reduced}")        
        print(f"[{batch}/{len_dataset}] total loss: {losses.item():.2f} losses: {loss_dict_printable}")     
        losses_avg+= losses.item()
        #if(batch+1)%2==0: break
        
    return losses_avg/len_dataset
        
        

def validation_one_epoch():      
    loss=0
    len_dataset=len(valiation_loader)  
    for  batch, data in enumerate(valiation_loader):
        images,targets=data            
        images=list(image.to(device) for image in images)   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            #predicts=model(images,None)                
            #predictions=reduce_dict(predictions[0])
            loss=loss+losses.item()
            #print(f"batch: {batch},validation loss:{losses.item():.2f}")  
            #print(f"predicts: {predicts}")        
        #if(batch+1)%2==0: break
    model.train()
    return loss/len_dataset
    

NUM_EPOCH=25
train_loss=[]
validation_loss=[]
for epoch in range(1,NUM_EPOCH+1):
    losses_avg=train_one_epoch()
    losses_avg_validation=validation_one_epoch()
    print(f"epoch[{epoch}/{NUM_EPOCH}]: avg. loss: {losses_avg}")
    print(f"epoch: {epoch},validation loss:{ losses_avg_validation:.2f}")  
    train_loss.append(losses_avg)
    validation_loss.append(losses_avg_validation)
    save_model(model)    
    writer.add_scalar('Segmentation/val_loss', losses_avg_validation, epoch)
    writer.add_scalar('Segmentation/train_loss', losses_avg, epoch)

print("train loss:\n")
print(validation_loss)

print("validation loss:\n")
print(validation_loss)    





