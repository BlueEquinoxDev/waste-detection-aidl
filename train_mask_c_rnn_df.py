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



data_transforms_train = transforms.Compose([        
    #transforms.Resize(size=(800,800)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor()
    ])

data_transforms_validation = transforms.Compose([ 
    #transforms.Resize(size=(800,800)),           
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor()])

def collate_fn(batch):
    images = []
    targets = []
    for i, t in batch:
        images.append(i)
        targets.append(t)
    return images, targets

train_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/train_annotations.json",
                                       img_dir="data",
                                       transforms=data_transforms_train)

validation_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/validation_annotations.json",
                                            img_dir="data",
                                            transforms=data_transforms_validation)


train_loader=DataLoader(train_taco_dataset,shuffle=True,batch_size=1,collate_fn=collate_fn)
valiation_loader=DataLoader(validation_taco_dataset,shuffle=True,batch_size=1,collate_fn=collate_fn)
 

from model.wastemaskrcnndf import MaskRCNNdf
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
backbone = resnet_fpn_backbone("resnet50", pretrained=True)
model = MaskRCNNdf(backbone)   
url = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
state_dict =  torch.hub.load_state_dict_from_url(url)
model.load_state_dict(state_dict)

num_classes = 28+1

# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                   hidden_layer,
                                                   num_classes)

model = model.train().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.0001)

model.train()
for i, (images, targets) in enumerate(train_loader):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    optimizer.zero_grad()
    predictions, loss_dict = model(images, targets)
    loss = sum(loss for loss in loss_dict.values())
    loss.backward()
    optimizer.step()
    if i%2 == 0:
        loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}
        print(f"[{i}/{len(train_loader)}] loss: {loss_dict_printable}")
    
        
checkpoint = {
        "model_state_dict":  model.cpu().state_dict(),
        "optimizer_state_dict":optimizer.state_dict()
}  

if not os.path.exists(f"{os.getcwd()}/app/checkpoint/"):
    os.makedirs(f"{os.getcwd()}/app/checkpoint/")
    
torch.save(checkpoint, f"{os.getcwd()}/app/checkpoint/checkpoint.pt")