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

device=torch.device("cpu")


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


