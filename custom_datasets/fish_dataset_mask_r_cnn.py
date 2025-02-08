from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image, ImageOps
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from torchvision import tv_tensors
import torch
import json
import numpy as np

class FishRCNNDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transforms=None):
        super().__init__()
        assert os.path.isfile(annotations_file), f"File not found: {annotations_file}"
        assert os.path.isdir(img_dir), f"Directory not found: {img_dir}"

        self.coco_data = COCO(annotations_file)
        self.len_dataset=len(self.coco_data.imgs.keys())
        self.img_dir = img_dir
        self.transforms = transforms
        #self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.index_to_imageId={i:img_id for i,img_id in enumerate(self.coco_data.imgs.keys())}
               
        # Fix this
        # Create mappings from the list of supercategory objects
        self.idx_to_class = {(item['id']): item['name'] 
                            for item in self.coco_data.loadCats(self.coco_data.getCatIds())}
        self.class_to_idx = {item['name']: (item['id']) 
                            for item in self.coco_data.loadCats(self.coco_data.getCatIds())}
        

    def __getitem__(self, idx):
        """im, boxes, labels, masks = shapes_dataset.random_image(100, 100)
        targets = {"boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                   "labels": torch.tensor(labels, dtype=torch.int64, device=device),
                   "masks": torch.tensor(np.stack(masks), dtype=torch.uint8, device=device)}
        return to_tensor(im).to(device), targets"""

        img_id=self.index_to_imageId[idx]
        img_coco_data = self.coco_data.loadImgs([img_id])[0]
        path = os.path.join(self.img_dir, img_coco_data['file_name'])
        annotations = self.coco_data.imgToAnns[img_id]
        sample_img = Image.open(path)
        # Avoid issues of rotated images by rotating it accoring to EXIF info
        sample_img = ImageOps.exif_transpose(sample_img)
        
        masks =[]
        bboxs=[]
        labels=[]
        for ann in annotations:
            masks.append(self.coco_data.annToMask(ann))
            bx=ann['bbox']
            bboxs.append([bx[0],bx[1],bx[0]+bx[2],bx[1]+bx[3]])
            labels.append(ann['category_id'])
        
        """
        print(f"self.device: {self.device}")
        targets = {"boxes": torch.tensor(bboxs, dtype=torch.float32, device=self.device),
                   "labels": torch.tensor(labels, dtype=torch.int64, device=self.device),
                   "masks": torch.tensor(np.stack(masks), dtype=torch.uint8, device=self.device)}
        
        sample_img = F.to_tensor(sample_img).to(self.device)
        """

        #print(f"boxes BEFORE target: {type(bboxs)}")

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(np.array(bboxs),
                                                  format="XYXY",
                                                   canvas_size=F.get_size(sample_img),
                                                   dtype=torch.float32, 
                                                   device=self.device)
        #target["boxes"] = torch.tensor(bboxs, dtype=torch.float32, device=self.device),

        target["labels"] = torch.tensor(labels,dtype=torch.int64, device=self.device)
        target["masks"] = tv_tensors.Mask(np.stack(masks),
                                        dtype=torch.uint8, 
                                        device=self.device)
        
        
        to_tensor = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
        sample_img = to_tensor(sample_img).to(self.device)

        #print(f"Target boxes BEFORE TRANSFORMS: {type(target['boxes'])}")

        if self.transforms is not None:
            sample_img, target = self.transforms(sample_img, target)
        
        #print(f"Target boxes AFTER TRANSFORMS: {type(target['boxes'])}")

        #print(f"sample_img.device: {sample_img.device}")
        #print(f"sample_img.shape: {sample_img.shape}")
        #print([f"{k}: {v}" for k, v in target.items()])

        
        return sample_img, target

    def __len__(self):
        return self.len_dataset