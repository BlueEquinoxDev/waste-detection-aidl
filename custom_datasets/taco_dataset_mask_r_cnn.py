from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image
from utilities.config_utils import TaskType, ClassificationCategoryType
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torch
import json
from PIL import ImageOps
import numpy as np

class TacoDatasetMaskRCNN(Dataset):

    def __init__(self,
                 annotations_file: str, 
                 img_dir: str, 
                 transforms=None) -> None:
        """ Constructor for the TacoDataset class """
        super().__init__()

        assert os.path.isfile(annotations_file), f"File not found: {annotations_file}"
        assert os.path.isdir(img_dir), f"Directory not found: {img_dir}"
        
        self.coco_data = COCO(annotations_file)
        self.len_dataset=len(self.coco_data.imgs.keys())
        self.img_dir = img_dir
        self.transforms = transforms
        
        self.index_to_imageId={i:img_id for i,img_id in enumerate(self.coco_data.imgs.keys())}
        
           # Load the new JSON with supercategories and their corresponding ids
        # Load supercategories from JSON
        with open('data/taco28_categories.json', 'r') as infile:
            supercategories_list = json.load(infile)
        
        # Create mappings from the list of supercategory objects
        self.idx2class = {item['id']+1: item['supercategory'] for item in supercategories_list}
        
        self.class2idx = {item['supercategory']: item['id']+1 for item in supercategories_list}
        
        # Create category mapping for COCO annotations
        self.category_map = {cat['id']: self.class2idx[cat['supercategory']] 
                             for cat in self.coco_data.loadCats(self.coco_data.getCatIds())}

    def __len__(self) -> None:
        return self.len_dataset
    
    def __getitem__(self, idx) -> None:

        img_id=self.index_to_imageId[idx]                        
        
        img_coco_data = self.coco_data.loadImgs([img_id])[0]

        path = os.path.join(self.img_dir, img_coco_data['file_name'])
        
        annotations = self.coco_data.imgToAnns[img_id]
        
        
        sample_img = Image.open(path)
        sample_img = ImageOps.exif_transpose(sample_img)
        
        sample_img = tv_tensors.Image(sample_img)
        
        masks =[]
        bboxs=[]
        areas=[]
        labels=[]
        for ann in annotations:
            masks.append(self.coco_data.annToMask(ann))
            bx=ann['bbox']
            bboxs.append([bx[0],bx[1],bx[0]+bx[2],bx[1]+bx[3]])
            areas.append(bx[2]*bx[3])
            labels.append(self.category_map[ann['category_id']])
        target = {}
        
        target["boxes"] = tv_tensors.BoundingBoxes(np.array(bboxs),
                                                   format="XYXY",
                                                   canvas_size=F.get_size(sample_img))
        #dtype=torch.float32, 
        #device=self.device)
                    
        target["masks"] = tv_tensors.Mask(np.stack(masks))
        #dtype=torch.uint8, 
        #device=self.device)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = idx
        #target["area"] =torch.tensor(areas)
        #target["iscrowd"] =torch.zeros( len(labels,), dtype=torch.int64)

        if self.transforms is not None:
            sample_img, target = self.transforms(sample_img, target)
        
        return sample_img, target
        