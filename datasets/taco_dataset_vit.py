from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image
#from utilities.config_utils import TaskType, ClassificationCategoryType
#from utilities.get_supercategory_by_id import get_supercategory_by_id
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
import json
from utilities.get_supercategory_by_id import get_supercategory_map

class TacoDatasetViT(Dataset):
    """
    Custom dataset for waste segmentation and classification using TACO dataset in COCO format
    
    params:
    - annotations_file: path to the annotations file
    - img_dir: path to the image directory
    - transforms: list of transformations to apply to the images
    - cls_category: classification category type (CATEGORY or SUPERCATEGORY)

    returns:
    In case of segmentation task:
    - sample_img: image numpy array
    - masks: numpy array with masks for each object in the image
    In case of classification task:
    - sample_img: image numpy array
    - category_id: category id
    """


    def __init__(self, annotations_file: str, img_dir: str, transforms=None) -> None:
        """ Constructor for the TacoDataset class """
        super().__init__()

        # Check if the provided paths are valid and if the task type is valid
        assert os.path.isfile(annotations_file), f"File not found: {annotations_file}"
        assert os.path.isdir(img_dir), f"Directory not found: {img_dir}"

        self.coco_data = COCO(annotations_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = list(self.coco_data.imgs.keys())
        self.anns_ids = list(self.coco_data.anns.keys())

        # Get all category IDs
        # cat_ids = self.coco_data.getCatIds()
        # Load categories using the IDs
        # categories = self.coco_data.loadCats(cat_ids)
        # Define a dictionary to map category IDs to category names
        # self.idx_to_class = {cat['id']: cat['name'] for cat in categories}
        # self.class_to_idx = {cat['name']: cat['id'] for cat in categories}

        # Load the new JSON with supercategories and their corresponding ids
        # Load supercategories from JSON
        with open('data/supercategories.json', 'r') as infile:
            supercategories_list = json.load(infile)
        
        # Create mappings from the list of supercategory objects
        self.idx_to_class = {item['id']: item['supercategory'] 
                            for item in supercategories_list}
        self.class_to_idx = {item['supercategory']: item['id'] 
                            for item in supercategories_list}
        # Create category mapping for COCO annotations
        self.category_map = {cat['id']: self.class_to_idx[cat['supercategory']] 
                        for cat in self.coco_data.loadCats(self.coco_data.getCatIds())}


    def __len__(self) -> None:
        """ Returns the length of the dataset """
        return len(self.img_ids)
    
    
    def square_img(self, width: int, height: int, img: np.ndarray) -> np.ndarray:
        """
        Make the image square by adding padding to the image
        params:
        - width: width of the image
        - height: height of the image
        - img: image to make square
        returns:
        - img: square image
        """
        # Check if the image is already square
        if width == height:
            return img
        # If the image is not square, calculate the padding needed
        elif width > height:
            delta = width - height
            pad_top = delta // 2
            pad_bottom = delta - pad_top
            pad_left, pad_right = 0, 0
        else:  # height > width:
            delta = height - width
            pad_left = delta // 2
            pad_right = delta - pad_left
            pad_top, pad_bottom = 0, 0
            
        # Add padding to make the image square
        return cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black padding
            )


    def __getitem__(self, idx) -> None:
        """ 
        Returns the sample and target (annotation) tensors at the given index 
        params:
        - idx: index of the image to retrieve
        returns:
        - sample: image tensor
        - target: annotation tensor
        """
        # Pick the annotation id based on given index
        # print("##################")
        # print(f"Index: {idx}")
        
        ann_id = self.anns_ids[idx]
        annotation = self.coco_data.loadAnns(ann_id)[0]
        bbox = annotation['bbox']  # it could be done using the bounding box instead of the segmentation
        category_id = self.category_map[annotation['category_id']]
        img_id = annotation['image_id']
        # print(f"Annotation id: {ann_id}")
        # print(f"Image id: {img_id}")
        # print("##################")
        # Load the image details using Coco API and image id
        img_coco_data = self.coco_data.loadImgs(img_id)[0] # The dict contains id, file_name, height, width, license and paths
        # Load path of the image using the image file name & Join the image directory path with the image file name
        path = os.path.join(self.img_dir, img_coco_data['file_name'])
        # Load the image using the path
        sample_img = Image.open(path)

        # Generate the mask from the annotation segmentation
        mask = self.coco_data.annToMask(annotation)
        # print(f"Mask shape: {mask.shape}")

        # Make the image a numpy array to apply the mask
        sample_img = np.array(sample_img)
        # print(f"Sample image shape: {sample_img.shape}")

        ### WORKAROUND ###
        rotated = False
        # if the mask is not the same size as the image, rotate it
        if mask.shape != sample_img.shape[:2]:
            mask = np.rot90(mask)
            rotated = True 

        # Apply the mask to the image
        cropped_image = cv2.bitwise_and(sample_img, sample_img, mask=mask)

        # Make square
        x_min, y_min, width, height = [int(dim) for dim in bbox]
        squared_img = self.square_img(width, height, cropped_image)
        
        # Convert to PIL Image for transforms
        # sample_img = Image.fromarray(sample_img)
        squared_img = np.array(squared_img)
        
        # Apply transforms
        if self.transforms:
          #   print("Applying transforms")
            squared_img = self.transforms(squared_img)

        # if rotated==True:
        #     print(f"Sample image shape: {squared_img.shape}")
        #     print(f"Sample image type: {squared_img.dtype}")
        #     print(f"Mask shape: {mask.shape}")
        #     print(f"Mask type: {mask.dtype}")
        #     # Convert tensor to numpy array and transpose dimensions for visualization
        #     if isinstance(sample_img, torch.Tensor):
        #         vis_img = sample_img.permute(1, 2, 0).numpy()  # Change from (C,H,W) to (H,W,C)
        #     else:
        #         vis_img = sample_img
                
        #     plt.figure(figsize=(20, 5))
        #     plt.subplot(1, 4, 1)
        #     plt.imshow(vis_img)
        #     plt.title('Original Image')
        #     plt.subplot(1, 4, 2)
        #     plt.imshow(mask, cmap='gray')
        #     plt.title('Mask')
        #     plt.subplot(1, 4, 3)
        #     if isinstance(cropped_image, torch.Tensor):
        #         cropped_vis = cropped_image.permute(1, 2, 0).numpy()
        #     else:
        #         cropped_vis = cropped_image
        #     plt.imshow(cropped_vis)
        #     plt.title('Masked Image')
        #     plt.subplot(1, 4, 4)
        #     if isinstance(squared_img, torch.Tensor):
        #         squared_vis = squared_img.permute(1, 2, 0).numpy()
        #     else:
        #         squared_vis = squared_img
        #     plt.imshow(squared_vis)
        #     plt.title('Squared and transformed Image')
        #     plt.show()

        return {
            'pixel_values': squared_img,
            'labels': category_id
        }

        
# TESTING THE DATASET

# data_transforms_train = transforms.Compose([
#     transforms.ToImage(),  # To tensor is deprecated
#     transforms.ToDtype(torch.uint8, scale=True),
#     transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
#     transforms.RandomRotation(degrees=15),
#     transforms.RandomHorizontalFlip(0.5), 
#     transforms.ToDtype(torch.float32, scale=True),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])


# taco_dataset = TacoDatasetViT(annotations_file='data/train_annotations.json', img_dir='data', transforms=data_transforms_train) 
# print(taco_dataset[37])