#see https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py#L67 for metrics
#https://stackoverflow.com/questions/79321438/how-to-train-mask-rcnn-mode-using-custom-dataset-with-pytorch
#https://www.kaggle.com/code/jyprojs/sartorius-torch-mask-r-cnn/notebook
#https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
from custom_datasets.taco_dataset_mask_r_cnn import TacoDatasetMaskRCNN
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import torch
import numpy as np
import os
import random
from pycocotools.cocoeval import COCOeval
from model.wastemaskrcnn import WasteMaskRCNN
from tqdm import tqdm

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

data_transforms_test = transforms.Compose([            
    transforms.Resize(size=(800,800)),                                           
    transforms.ToDtype(torch.float32, scale=True),
    transforms.ToPureTensor()])

test_taco_dataset=TacoDatasetMaskRCNN(annotations_file="data/test_annotations28.json",
                                      img_dir="data/images",
                                      transforms=data_transforms_test)


checkpoint_path = "results/mask_rcnn/checkpoint_epoch_6_2025_2_26_21_30.pt"
checkpoint = torch.load(checkpoint_path)


from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 

model = WasteMaskRCNN(num_classes=len(test_taco_dataset.idx2class))   

model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)



def coco_result_format(image_id:str,prediction:torch.Tensor,threshold:float = 0.5)->np.ndarray:
    #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #https://cocodataset.org/#format-results
    #https://cocodataset.org/#detection-eval
        
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


def collate_fn(batch):
    return tuple(zip(*batch))

test_loader=DataLoader(test_taco_dataset,
                            shuffle=True,
                            batch_size=1,
                            collate_fn=lambda batch: tuple(zip(*batch)))
 
results_bbox=[]
results_masks=[]
images_not_predicts=[]
def validation_one_epoch():      
    pbar = tqdm(test_loader, desc="Computing metrics test dataset", leave=False)
    for batch_idx, data in enumerate(pbar):
    #for  batch, data in enumerate(test_loader):
        model.train()
        images,targets=data            
        images=list(image.to(device) for image in images)   
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():                        
            predicts=model(images,None)                
        
        coco_result_format(targets[0]['image_id'],predicts[0])
        
    print("coco metrics for masks:\n")            
    coco_result=test_loader.dataset.coco_data.loadRes(results_masks)
    coco_eval=COCOeval(test_loader.dataset.coco_data,coco_result,"segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()    
    print("\n\n")
    
    print("coco metrics for bbox:\n")            
    coco_result=test_loader.dataset.coco_data.loadRes(results_bbox)
    coco_eval=COCOeval(test_loader.dataset.coco_data,coco_result,"bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
validation_one_epoch()


        
        


