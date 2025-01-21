#see https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py#L67 for metrics
from datasets.taco_dataset_mask_r_cnn import TacoDatasetMaskRCNN
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import torch
import numpy as np
import os
import random
from pycocotools.cocoeval import COCOeval
from model.wastemaskrcnn import WasteMaskRCNN

sed=23
torch.manual_seed(sed)
random.seed(sed)
np.random.seed(sed)

#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def compute_mean_std():
    taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/train_annotations.json",
                                     img_dir="data",
                                     transforms=transforms.Compose([
                                         transforms.ToDtype(torch.float32, scale=True),
                                         transforms.ToPureTensor()]))
    loader=DataLoader(train_taco_dataset,shuffle=False,batch_size=1)
    
    mean=[]
    std=[]
    for  batch, data in enumerate(loader):
        img,target=data
        img=img/255
        mean.append(torch.mean(img,dim=[1,2]).detach().cpu().numpy())
        std.append(torch.std(img,dim=[1,2]).detach().cpu().numpy())

    print(f"mean: ",np.mean(mean,axis=0))
    print(f"std: ",np.std(mean,axis=0))
    print('hi')
    exit(0)


data_transforms_train = transforms.Compose([        
    transforms.Resize(size=(800,800)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    transforms.Normalize(mean=[0.49515063, 0.46845073, 0.4139734], std=[0.11767369, 0.1058425 , 0.11804706])
    ])

data_transforms_validation = transforms.Compose([        
    transforms.Resize(size=(800,800)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    transforms.Normalize(mean=[0.49515063, 0.46845073, 0.4139734], std=[0.11767369, 0.1058425 , 0.11804706])
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
 

model=WasteMaskRCNN(num_classes=61)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
print(f"parameters to optimize: {len(params)}")
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

scaler = torch.cuda.amp.GradScaler()  if device.type == 'cuda' else None

def train_one_epoch():    
    for  batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        images,targets=data    
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        #loss_dict = model(images,targets)
        
        #with torch.autocast(device_type="cuda"):
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        #loss_dict_reduced = reduce_dict(loss_dict)       
        #losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        #loss_value = losses_reduced.item()
        #loss_dict_reduced={key:loss_dict_reduced[key].item() for key in  loss_dict_reduced.keys()}
        
        if scaler:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            old_scaler = scaler.get_scale()
            scaler.update()
            new_scaler = scaler.get_scale()
            if new_scaler >= old_scaler:
                lr_scheduler.step()
        else:
            losses.backward()
            optimizer.step()
            lr_scheduler.step()
        
        #print(f"batch: {batch}, loss:{loss_value} losses: {loss_dict_reduced}")
        print(f"batch: {batch}, loss:{losses.item()}")                
        
        

def validation_one_epoch():      
    #device = torch.device("cpu")
    #model.to(device)    
    #model.eval()    
    for  batch, data in enumerate(valiation_loader):        
        images,targets=data            
        images=list(image.to(device) for image in images)   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                #if torch.cuda.is_available():
                    #torch.cuda.synchronize()
                loss_dict = model(images, targets)      
                losses = sum(loss for loss in loss_dict.values())
                
                #predictions=reduce_dict(predictions[0])
                print(f"batch: {batch},validation loss:{losses.item()}")  


NUM_EPOCH=2
for epoch in range(NUM_EPOCH):
    train_one_epoch()
    validation_one_epoch()

checkpoint = {
        "model_state_dict":  model.cpu().state_dict(),
        "optimizer_state_dict":optimizer.state_dict()
}  

if not os.path.exists(f"{os.getcwd()}/app/checkpoint/"):
    os.makedirs(f"{os.getcwd()}/app/checkpoint/")
    
torch.save(checkpoint, f"{os.getcwd()}/app/checkpoint/checkpoint.pt")
