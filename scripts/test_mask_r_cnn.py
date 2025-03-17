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
import argparse

from pycocotools.mask import encode
from pycocotools.mask import decode
from pycocotools.mask import iou
from pycocotools.mask import area
from pycocotools.mask import toBbox
from pycocotools.mask import frPyObjects
from typing import List

import torch
from torchvision.ops import nms
from utilities.compute_metrics import compute_iou

from torchmetrics.detection import CompleteIntersectionOverUnion
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection import MeanAveragePrecision

from utilities.compute_metrics import FalseNegativesRate
from utilities.compute_metrics import FalsePositivesRate

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
test_loader=DataLoader(test_taco_dataset,
                        shuffle=False,
                        batch_size=1,
                        collate_fn=lambda batch: tuple(zip(*batch)))



def mask_to_coco_format(threshold,mask):
        a=mask.cpu().squeeze().numpy()
        a=(a>=threshold).astype(np.uint8)
        a=np.asfortranarray(a)
        return encode(a)
 
 
def bbox_to_coco_format(bbox:torch.Tensor)->None:
    bbox=bbox.cpu().numpy()
    return [bbox[0],bbox[1],(bbox[2]-bbox[0]),(bbox[3]-bbox[1])]
 
def post_processing(prediction,score_threshold=0.4,iou_threshold = 0.2):
 
    p=prediction.copy()
 
    #Step 1: remove background
    valid_indices = torch.where(p['labels'] != 0)[0]
    p['scores'] =p['scores'][valid_indices]
    p['boxes']=p['boxes'][valid_indices]
    p['labels']=p['labels'][valid_indices]
    p['masks']=p['masks'][valid_indices]
 
    # Step 2: Apply score threshold
    valid_indices = torch.where(p['scores'] >= score_threshold)[0]
    p['scores'] =p['scores'][valid_indices]
    p['boxes']=p['boxes'][valid_indices]
    p['labels']=p['labels'][valid_indices]
    p['masks']=p['masks'][valid_indices]
 
    # Step 3: Non-Maximum Suppression 
    indices=nms(p['boxes'],p['scores'],iou_threshold)
    p['boxes']=p['boxes'][indices]
    p['labels']=p['labels'][indices]
    p['masks']=p['masks'][indices]
    return p
 
 
def coco_result_format(images_id:List[int], predictions:List[torch.Tensor],threshold:float = 0.4)->np.ndarray:
    #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #https://cocodataset.org/#format-results
    #https://cocodataset.org/#detection-eval    
    results=[]
    for i in range(len(images_id)):
        image_id=images_id[i]
        prediction=post_processing(predictions[i],threshold)
        for i in range(len(prediction['labels'])):
            results.append({
                    "image_id":image_id,
                    "category_id":prediction['labels'][i].item(),
                    "bbox":bbox_to_coco_format(prediction['boxes'][i]),                
                    "score":prediction['scores'][i].item(),
                    "segmentation":mask_to_coco_format(threshold,prediction['masks'][i])})     
    return results
 
def image_show(img,detections):
    named_labels = [v for k, v in test_taco_dataset.idx2class.items()]    
    image = torchvision.transforms.functional.to_pil_image(img)
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
 
def validation_one(predicts,targets):
    images_id=[t['image_id'] for t in targets]                
    results=coco_result_format(images_id,predicts)
 
    print("coco metrics for masks:\n")            
    coco_result=test_loader.dataset.coco_data.loadRes(results)
    coco_eval=COCOeval(test_loader.dataset.coco_data,coco_result)    
    coco_eval.params.useCats=0    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()    
    print("\n\n")
 
def show_me_one(idx):        
    len_dataset=len(test_loader.dataset)
 
 
    img,target=test_taco_dataset[idx]
    img=img.to(device)
    target =[{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in target.items() }]
 
    model.eval()    
    with torch.no_grad():          
        detections=model([img])    

    detections=post_processing(detections[0])
 
    image_show(img,detections)
 
 
 
def show_me_all():        
    named_labels = [v for k, v in test_taco_dataset.idx2class.items()]
    model.eval()
    for  batch, data in enumerate(test_loader):        
        images,targets=data            
        images=list(image.to(device) for image in images)      
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]     
        with torch.no_grad():          
            detections=model(images)              
 
        detections=post_processing(detections[0])           
        image_show(images[0],detections)
 
def mask2bin(predicts):
    for pred in predicts:
        pred['masks'] = (pred['masks'].squeeze(1) > 0.5).byte()  
    return predicts                                   
 
def validation():      
    ioU_no_respect_labels_avg=0
    ioU_respect_labels_avg=0
    cIoU_no_respect_labels_avg=0
    cIoU_respect_labels_avg=0
    fp_rate_avg=0
    fn_rate_avg=0
    all_results=[]
 
    cIoU_no_respect_labels=CompleteIntersectionOverUnion(box_format='xyxy',respect_labels=False).to(device)
    cIoU_respect_labels=CompleteIntersectionOverUnion(box_format='xyxy',respect_labels=True).to(device)
 
    ioU_no_respect_labels=IntersectionOverUnion(box_format='xyxy',respect_labels=False).to(device)
    ioU_respect_labels=IntersectionOverUnion(box_format='xyxy',respect_labels=True).to(device)
 
 
    mAP_test = MeanAveragePrecision(iou_type=["bbox",'segm'])
    mAP_test.warn_on_many_detections=False
 
    len_dataset=len(test_loader.dataset)
 
    model.eval()
    pbar = tqdm(test_loader, desc="Computing metrics test dataset", leave=False)
    for batch_idx, data in enumerate(pbar):    
        images,targets=data            
        images=list(image.to(device) for image in images)   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():                        
            predicts=model(images,None)                        
 
 
 
        cIoU_no_respect_labels_avg+=cIoU_no_respect_labels(predicts,targets)['ciou'].item()
        cIoU_respect_labels_avg+=cIoU_respect_labels(predicts,targets)['ciou'].item()
 
        ioU_no_respect_labels_avg+=ioU_no_respect_labels(predicts,targets)['iou'].item()
        ioU_respect_labels_avg+=ioU_respect_labels(predicts,targets)['iou'].item()
 
        fp_rate_avg+=FalsePositivesRate(predicts[0]['labels'],targets[0]['labels'])
        fn_rate_avg+=FalseNegativesRate(predicts[0]['labels'],targets[0]['labels'])
 
        mAP_test.update(mask2bin(predicts),targets)    
 
        images_id=[t['image_id'] for t in targets]                
        results=coco_result_format(images_id,predicts)
        if (len(results)>0): 
            if len(all_results)>0:all_results=all_results+results
            else:all_results=results
 
 
    print("Metrics:")
    print(f"Avg. IoU (respect labels):{ioU_respect_labels_avg/len_dataset:.2f}")
    print(f"Avg. IoU (no respect labels):{ioU_no_respect_labels_avg/len_dataset:.2f}")
    print(f"Avg. Complete IoU (respect labels):{cIoU_respect_labels_avg/len_dataset:.2f}")
    print(f"Avg. Complete IoU (no respect labels):{cIoU_no_respect_labels_avg/len_dataset:.2f}")
    print(f"Avg. False positives rate:{fp_rate_avg/len_dataset:.2f}")
    print(f"Avg. False negatives rate:{fn_rate_avg/len_dataset:.2f}")
 
    mAP_test_results = mAP_test.compute()
    for metric_name, metric_value in mAP_test_results.items() :
        if metric_name!='classes':
            print(f"{metric_name}: {metric_value.item():.2f}")
    print(f"predicted classes: {[l.item() for l in mAP_test_results['classes']]}")
 
 
    print("coco metrics for masks:")            
    coco_result=test_loader.dataset.coco_data.loadRes(all_results)
    coco_eval=COCOeval(test_loader.dataset.coco_data,coco_result)        
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()    
    print("\n\n")
 
 
 
parser = argparse.ArgumentParser(description='Test Mask r-cnn ober TACO 28 dataset')
parser.add_argument('--checkpoint_path', required=True, help=f'Checkpoint path to compute the metrics',type=str)
parser.add_argument('--show_image', required=False, help=f'Show prediction of a image take on taco test dataset (1,{len(test_taco_dataset)})',type=int)
 
args = parser.parse_args()          
 
checkpoint_path =args.checkpoint_path # "results/mask_rcnn/checkpoint_epoch_22_2025_3_15_18_38.pt"
checkpoint = torch.load(checkpoint_path)
 
 
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
 
model = WasteMaskRCNN(num_classes=len(test_taco_dataset.idx2class))   

model.load_state_dict(checkpoint['model_state_dict'])
                   
model.to(device)          
 
if args.show_image:
    assert args.show_image>0 and args.show_image<(len(test_taco_dataset)+1)
    show_me_one(args.show_image-1)
else:
    validation()
