#see https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py#L67 for metrics
#https://stackoverflow.com/questions/79321438/how-to-train-mask-rcnn-mode-using-custom-dataset-with-pytorch
#https://www.kaggle.com/code/jyprojs/sartorius-torch-mask-r-cnn/notebook
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


data_transforms_test = transforms.Compose([            
    #transforms.Resize(size=(800,800)),                                           
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor()])

test_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/test_annotations.json",
                                      img_dir="data",
                                      transforms=data_transforms_test)


checkpoint_path = "app/checkpoint/checkpoint.pt"
checkpoint = torch.load(checkpoint_path)


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
#backbone = resnet_fpn_backbone("resnet50", pretrained=True)
model = WasteMaskRCNN(num_classes=29)   

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

'''
image,target=test_taco_dataset[56]

image=image.to(device)

model.eval()
with torch.no_grad():
    detections = model([image], targets=None)    


print('targets:\n')
print(target)

print('predictions:\n')
print(detections)

print('hi')
'''


data_transforms_validation = transforms.Compose([            
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

validation_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/validation_annotations.json",
                                            img_dir="data",
                                            transforms=data_transforms_validation)

def collate_fn(batch):
    return tuple(zip(*batch))

valiation_loader=DataLoader(validation_taco_dataset,
                            shuffle=True,
                            batch_size=1,
                            collate_fn=lambda batch: tuple(zip(*batch)))
 
 
images_not_predicts=[]
def validation_one_epoch():      
    for  batch, data in enumerate(valiation_loader):
        model.train()
        images,targets=data            
        images=list(image.to(device) for image in images)   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():            
            loss_dict = model(images, targets)
            losses = sum([loss for loss in loss_dict.values()])
            predicts=model(images,None)                
            #predictions=reduce_dict(predictions[0])
            print(f"batch: {batch},validation loss:{losses.item():.2f}")  
            print(f"predicts: {predicts}")        
        if len(predicts[0]["labels"])==0: 
            images_not_predicts.append({"image_id":targets[0]['image_id'],"loss":losses.item()})
    print(images_not_predicts)
validation_one_epoch()
        
        


