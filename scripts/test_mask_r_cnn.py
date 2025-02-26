#see https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py#L67 for metrics
from datasets.taco_dataset_mask_r_cnn import TacoDatasetMaskRCNN
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import torch
import numpy as np
import os
import random
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import encode
from pycocotools.mask import decode
from pycocotools.mask import iou
from pycocotools.mask import area
from pycocotools.mask import toBbox
from pycocotools.mask import frPyObjects


sed=23
torch.manual_seed(sed)
random.seed(sed)
np.random.seed(sed)

#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def coco_result_format(image_id:str,prediction:torch.Tensor,threshold:float = 0.5)->np.ndarray:
    #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #https://cocodataset.org/#format-results
    #https://cocodataset.org/#detection-eval
    results_bbox=[]
    results_masks=[]
    
    def mask_to_coco_format(mask):
        a=mask.cpu().squeeze().numpy()
        a=(a>=threshold).astype(np.uint8)
        a=np.asfortranarray(a)
        return encode(a)

    def bbox_to_coco_format(bbox:torch.Tensor)->np.ndarray:
        bbox=bbox.cpu().numpy()
        return [bbox[0],bbox[1],(bbox[2]-bbox[0]),(bbox[3]-bbox[1])]
    
    for i in range(len(prediction['labels'])):            
        results_bbox.append({
                "image_id":image_id,
                "category_id":prediction['labels'][i].item(),
                "bbox":bbox_to_coco_format(prediction['boxes'][i]),                
                "score":prediction['scores'][i].item()})
        
        results_masks.append({
                "image_id":image_id,
                "category_id":prediction['labels'][i].item(),                
                "segmentation":mask_to_coco_format(prediction['masks'][i]),
                "score":prediction['scores'][i].item()})
        
    return results_bbox,results_masks


data_transforms_test = transforms.Compose([            
    #transforms.Resize(size=(800,800)),                                           
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor()
    ])

test_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/test_annotations.json",
                                      img_dir="data",
                                      transforms=data_transforms_test)


checkpoint_path = "app/checkpoint/checkpoint.pt"
checkpoint = torch.load(checkpoint_path)


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
backbone = resnet_fpn_backbone("resnet50", pretrained=True)
model = MaskRCNNdf(backbone,num_classes=29)   

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)

image,target=test_taco_dataset[80]

image=image.to(device)

model = model.eval()
with torch.no_grad():
    detections, loss_dict = model([image], targets=None)


print('targets:\n')
print(target)

print('predictions:\n')
print(detections)

print('hi')


