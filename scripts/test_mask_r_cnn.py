#see https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py#L67 for metrics
#https://stackoverflow.com/questions/79321438/how-to-train-mask-rcnn-mode-using-custom-dataset-with-pytorch
#https://www.kaggle.com/code/jyprojs/sartorius-torch-mask-r-cnn/notebook
#https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
from custom_datasets.taco_dataset_mask_r_cnn import TacoDatasetMaskRCNN
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import torchvision
import torch
import numpy as np
import os
import random
from pycocotools.cocoeval import COCOeval
from model.wastemaskrcnn import WasteMaskRCNN
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps

from pycocotools.mask import encode
from pycocotools.mask import decode
from pycocotools.mask import iou
from pycocotools.mask import area
from pycocotools.mask import toBbox
from pycocotools.mask import frPyObjects
from typing import List

import torch
from torchvision.ops import nms

sed=23
torch.manual_seed(sed)
random.seed(sed)
np.random.seed(sed)

#can cause differents algorithms in subsecuents runs 
# (False reduce performance but use the same algorithm always)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

data_transforms_test = transforms.Compose([            
    transforms.Resize(size=(800,800)),                                           
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor()])

test_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/test_annotations.json",
                                      img_dir="data/images",
                                      transforms=data_transforms_test)


checkpoint_path = "results/mask_rcnn/checkpoint_epoch_6_2025_2_26_21_30.pt"
checkpoint = torch.load(checkpoint_path)


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 

model = WasteMaskRCNN(num_classes=len(test_taco_dataset.idx2class))   

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)


def mask_to_coco_format(threshold,mask):
        a=mask.cpu().squeeze().numpy()
        a=(a>=threshold).astype(np.uint8)
        a=np.asfortranarray(a)
        return encode(a)
        

def bbox_to_coco_format(bbox:torch.Tensor)->None:
    bbox=bbox.cpu().numpy()
    return [bbox[0],bbox[1],(bbox[2]-bbox[0]),(bbox[3]-bbox[1])]

def apply_non_maximun_suppresion(prediction,score_threshold=0.4,iou_threshold = 0.5):
    

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


def coco_result_format(images_id:List[int], predictions:List[torch.Tensor],threshold:float = 0.4)->np.ndarray:
    #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #https://cocodataset.org/#format-results
    #https://cocodataset.org/#detection-eval
        
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
        


def collate_fn(batch):
    return tuple(zip(*batch))

test_loader=DataLoader(test_taco_dataset,
                            shuffle=False,
                            batch_size=1,
                            collate_fn=lambda batch: tuple(zip(*batch)))


def validation_one(predicts,targets):
    images_id=[t['image_id'] for t in targets]                
    coco_result_format(images_id,predicts)
    
    print("coco metrics for masks:\n")            
    coco_result=test_loader.dataset.coco_data.loadRes(results)
    coco_eval=COCOeval(test_loader.dataset.coco_data,coco_result)    
    coco_eval.params.useCats=0    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()    
    print("\n\n")
    

def show_me():        
    named_labels = [v for k, v in test_taco_dataset.idx2class.items()]
    for  batch, data in enumerate(test_loader):
        model.train()
        images,targets=data            
        images=list(image.to(device) for image in images)           
        with torch.no_grad():          
            detections=model(images)              
            detections=apply_non_maximun_suppresion(detections[0])
            image = torchvision.transforms.functional.to_pil_image(images[0])
            im_copy = image.copy()
            #print(im_copy)
            for box, score, label, mask in zip(detections['boxes'], detections['scores'], detections['labels'], detections['masks']):                
                mask = (mask[0].cpu().detach().numpy()*128).astype(np.int8)
                mask_im = Image.fromarray(mask, mode="L")
                full_color = Image.new("RGB", im_copy.size, (0, 255, 0))
                im_copy = Image.composite(full_color, im_copy, mask_im)
                coords = box.cpu().tolist()
                draw = ImageDraw.Draw(im_copy)
                draw.rectangle(coords, width=1, outline=(0, 255, 0))
                text = f"{named_labels[label-1]} {score*100:.2f}%"
                draw.text([coords[0], coords[1]-20], text, fill=(0, 255, 0), font_size=20)
            print(detections['labels'])

            im_copy.show()
            

            
                    
                
 
results=[]
def validation_one_epoch():      
    pbar = tqdm(test_loader, desc="Computing metrics test dataset", leave=False)
    for batch_idx, data in enumerate(pbar):
    #for  batch, data in enumerate(test_loader):
        model.train()
        images,targets=data            
        images=list(image.to(device) for image in images)   
        #targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():                        
            predicts=model(images,None)                
        #validation_one(predicts,targets)
        images_id=[t['image_id'] for t in targets]                
        coco_result_format(images_id,predicts)
        
    print("coco metrics for masks:\n")            
    coco_result=test_loader.dataset.coco_data.loadRes(results)
    coco_eval=COCOeval(test_loader.dataset.coco_data,coco_result)    
    coco_eval.params.useCats=0    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()    
    print("\n\n")
        
#show_me()
validation_one_epoch()


        
        


