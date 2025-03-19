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
import numpy as np
from utilities.earlystoping import EarlyStopping
from utilities.compute_metrics import FalseNegativesRate
from utilities.compute_metrics import FalsePositivesRate
from utilities.save_model import save_model


from torchmetrics.detection import CompleteIntersectionOverUnion
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection import MeanAveragePrecision

from tqdm import tqdm

h_params ={
    "batch_size": 1,
    "num_workers": 0,
    "mask_hidden_layer": 256,
    "aspect_ratio":(0.5, 1.0, 2.0),
    "anchor_sizes":((16,), (32,), (64,), (256,), (512,)),
    "box_detections_per_img":256
}

sed=23
torch.manual_seed(sed)
random.seed(sed)
np.random.seed(sed)

#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


logdir = os.path.join("logs", f"segmentation-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

# TODO: Initialize Tensorboard Writer with the previous folder 'logdir'
writer=SummaryWriter(log_dir=logdir)

# Create the datasets for train and validate
data_transforms_train = transforms.Compose([            
transforms.Resize((800,800)),    
    transforms.ColorJitter(brightness=0.5, contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=0.05),  
    transforms.RandomChoice([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5)]),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation= transforms.InterpolationMode.BILINEAR, fill=0),
    transforms.RandomChoice([transforms.RandomRotation(90, interpolation= transforms.InterpolationMode.NEAREST , expand=True, center=None, fill=0),
                             transforms.RandomRotation(180, interpolation= transforms.InterpolationMode.NEAREST, expand=True, center=None, fill=0),
                             transforms.RandomRotation(270, interpolation= transforms.InterpolationMode.NEAREST, expand=True, center=None, fill=0)]),    
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    ])

data_transforms_validation = transforms.Compose([
    transforms.Resize((800,800)),    
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    #transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
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

#Computing samples weights for WeightedRandomSampler
number_of_samples_by_category={cat:len(train_taco_dataset.coco_data.catToImgs[cat]) for cat in train_taco_dataset.idx2class.keys() if cat!=0 }
number_of_samples=sum({cat:len(train_taco_dataset.coco_data.catToImgs[cat]) for cat in train_taco_dataset.idx2class.keys() if cat!=0}.values())
category_weights={cat:number_of_samples/value for cat,value in number_of_samples_by_category.items()}
samples_weights=[]

for idx in range(train_taco_dataset.len_dataset):
    weights_cat_in_image=[]
    img_id=train_taco_dataset.index_to_imageId[idx]                        
    img_coco_data = train_taco_dataset.coco_data.loadImgs([img_id])[0]
    annotations = train_taco_dataset.coco_data.imgToAnns[img_id]
    for ann in annotations:
        weights_cat_in_image.append(category_weights[ann['category_id']])
    samples_weights.append(max(weights_cat_in_image))

sampler=torch.utils.data.WeightedRandomSampler(weights=samples_weights, num_samples=train_taco_dataset.len_dataset, replacement=True)

# Generate Dataloaders for train & validation
train_loader=DataLoader(train_taco_dataset,
                        shuffle=True,
                        batch_size=h_params['batch_size'],
                        num_workers=h_params["num_workers"],
                        collate_fn=collate_fn,
                        sampler=None)

valiation_loader=DataLoader(validation_taco_dataset,
                            shuffle=False,
                            batch_size=h_params['batch_size'],
                            num_workers=h_params["num_workers"],
                            collate_fn=collate_fn)
 

# Import model and assign the nr of classes
model=WasteMaskRCNN(len(train_taco_dataset.idx2class))
model.to(device)

#print params to optimize
num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f"params to optimize: {num_params}")



# Create the optimizer
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-2)

# Define the scheduler
scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5)



# Save checkpoint function to pass a early stopping if we want to do
def save_checkpoint(model, epoch):
    save_results = os.path.join("results","mask_rcnn")
    save_model(model,epoch,optimizer,idx2class,save_results)

# Define the early stopping    
early_stopping = EarlyStopping(patience=5, delta=0.05,path_dir_save_checkpoint=os.path.join("results","mask_rcnn"))


#train function for one epoch
def train_one_epoch():      
    """
    Function to train 1 epoch of Mask R-CNN
    returns:
        -  avg losses in training
    """
    model.train()
    losses_avg=0
    loss_mask_avg=0
    loss_box_reg_avg=0
    loss_rpn_box_reg_avg=0
    loss_objectness_avg=0
    len_dataset=len(train_loader)  
    for  batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        images,targets=data    
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
        print(f"[{batch}/{len_dataset}] total loss: {losses.item():.2f} losses: {loss_dict_printable}")     
        losses_avg+= losses.item()
        loss_mask_avg+=loss_dict['loss_mask'].item()
        loss_box_reg_avg+=loss_dict['loss_box_reg'].item()
        loss_rpn_box_reg_avg+=loss_dict['loss_rpn_box_reg'].item()
        loss_objectness_avg+=loss_dict['loss_objectness'].item()
        
        #if(batch+1)%10==0: break
        
    return {
        "loss":losses_avg/len_dataset, 
        "loss_mask":loss_mask_avg/len_dataset,
        "loss_box_reg":loss_box_reg_avg/len_dataset,
        "loss_rpn_box":loss_rpn_box_reg_avg/len_dataset,
        "loss_objectness":loss_objectness_avg/len_dataset}
        

#define metrics:        
cIoU_no_respect_labels=CompleteIntersectionOverUnion(box_format='xyxy',respect_labels=False).to(device)
cIoU_respect_labels=CompleteIntersectionOverUnion(box_format='xyxy',respect_labels=True).to(device)

ioU_no_respect_labels=IntersectionOverUnion(box_format='xyxy',respect_labels=False).to(device)
ioU_respect_labels=IntersectionOverUnion(box_format='xyxy',respect_labels=True).to(device)


#validation function for one epoch
def validation_one_epoch():  
    """
    Function to validate 1 epoch of Mask R-CNN
    returns:
        -  avg losses and metrics in validation dataset
    """    
    def mask2bin(predicts):
        for pred in predicts:
            pred['masks'] = (pred['masks'].squeeze(1) > 0.5).byte()  
        return predicts 

    num_no_predictions=0
    scores_avg=0
    loss=0    
    loss_mask_avg=0
    loss_box_reg_avg=0
    loss_rpn_box_reg_avg=0
    loss_objectness_avg=0
    fp_rate_avg=0
    fn_rate_avg=0
    ioU_no_respect_labels_avg=0
    ioU_respect_labels_avg=0
    cIoU_no_respect_labels_avg=0
    cIoU_respect_labels_avg=0
    
    mAP_test = MeanAveragePrecision(iou_type="segm")
    mAP_test.warn_on_many_detections=False
    
    len_dataset=len(valiation_loader)  
    pbar = tqdm(valiation_loader, desc="Computing loss for validation dataset", leave=False)
    for batch_idx, data in enumerate(pbar):
        images,targets=data            
        images=list(image.to(device) for image in images)   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            predicts=model(images,None)                
        model.train()
        loss=loss+losses.item()
        loss_mask_avg+=loss_dict['loss_mask'].item()
        loss_box_reg_avg+=loss_dict['loss_box_reg'].item()
        loss_rpn_box_reg_avg+=loss_dict['loss_rpn_box_reg'].item()
        loss_objectness_avg+=loss_dict['loss_objectness'].item()
        
        fp_rate_avg+=FalsePositivesRate(predicts[0]['labels'],targets[0]['labels'])
        fn_rate_avg+=FalseNegativesRate(predicts[0]['labels'],targets[0]['labels'])
        
        cIoU_no_respect_labels_avg+=cIoU_no_respect_labels(predicts,targets)['ciou'].item()
        cIoU_respect_labels_avg+=cIoU_respect_labels(predicts,targets)['ciou'].item()

        ioU_no_respect_labels_avg+=ioU_no_respect_labels(predicts,targets)['iou'].item()
        ioU_respect_labels_avg+=ioU_respect_labels(predicts,targets)['iou'].item()
        
        mAP_test.update(mask2bin(predicts),targets)    
            
        if(len(predicts[0]['labels'])==0):
            num_no_predictions+=1            
        else:
            scores_avg+=np.average([s.cpu() for s in  predicts[0]['scores']])
            
        #print(f"predicts: {predicts}")        
        #if(batch_idx+1)%10==0: break    
        
    avg_losses={
        "loss":loss/len_dataset, 
        "loss_mask":loss_mask_avg/len_dataset, 
        "loss_box_reg":loss_box_reg_avg/len_dataset, 
        "loss_rpn_box_reg":loss_rpn_box_reg_avg/len_dataset,
        "loss_objectness":loss_objectness_avg/len_dataset}
    avg_metrics={
        "num_no_predictions":num_no_predictions/len_dataset, 
        "scores":scores_avg /len_dataset, 
        "False Positives Rate":fp_rate_avg/len_dataset, 
        "False Negative Rate":fn_rate_avg/len_dataset,
        "ioU_no_respect_labels_avg":ioU_no_respect_labels_avg/len_dataset,
        "ioU_respect_labels_avg":ioU_respect_labels_avg/len_dataset,
        "cIoU_no_respect_labels_avg":cIoU_no_respect_labels_avg/len_dataset,
        "cIoU_respect_labels_avg":cIoU_respect_labels_avg/len_dataset}
    
    mAP_test_results = mAP_test.compute()
    for metric_name, metric_value in mAP_test_results.items() :
        if metric_name!='classes':
            avg_metrics[metric_name]=metric_value.item()            
    return avg_losses,avg_metrics
    

#train bucle
NUM_EPOCH=25
train_loss=[]
validation_loss=[]
for epoch in range(1,NUM_EPOCH+1):
    train_losses=train_one_epoch()
    avg_losses,avg_metrics=validation_one_epoch()
    scheduler.step(avg_losses["loss"])
    
    print(f"epoch[{epoch}/{NUM_EPOCH}]: avg. loss: {train_losses['loss']}")
    print(f"epoch[{epoch}/{NUM_EPOCH}]: validation avg. loss:{avg_losses['loss']:.2f}")  
    
    train_loss.append(train_losses['loss'])
    validation_loss.append(avg_losses["loss"])
    
    if early_stopping.fn_save_checkpoint==None:save_checkpoint(model,epoch)    
    early_stopping(avg_losses["loss"],model,epoch)
    
    print(f"\nSummary [{epoch}/{NUM_EPOCH}]:")
    print(f"\tTrain loss:")  
    for k,v in train_losses.items():
        writer.add_scalar(f'Segmentation_loss/train_{k}', v, epoch)
        print(f"\t\t{k}: {v:.2f}")  
    
    print(f"\tValidation loss:")  
    for k,v in avg_losses.items():
        writer.add_scalar(f'Segmentation_loss/val_{k}', v, epoch)
        print(f"\t\t{k}: {v:.2f}")  

    print(f"\tValidation metric:")  
    for k,v in avg_metrics.items():
        writer.add_scalar(f'Segmentation_val_metrics/val_{k}', v, epoch)
        print(f"\t\t{k}: {v:.2f}")  

    
    for name, weight in model.named_parameters():
        if weight.requires_grad==True:
            writer.add_histogram(f"Segmentation/{name}/value",weight,epoch)
            
    #stop if validation error don't change 
    if early_stopping.stop:
        print(f"early stopping at epoch {epoch}")
        break

#print the loss error if we want to plot or see
print("train loss:\n")
print(validation_loss)

print("validation loss:\n")
print(validation_loss)    





