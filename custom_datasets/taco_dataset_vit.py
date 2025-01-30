from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image, ImageOps
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


    def __init__(self, annotations_file, img_dir, transforms=None, subset_classes=None):
        super().__init__()
        
        # Check paths
        assert os.path.isfile(annotations_file), f"File not found: {annotations_file}"
        assert os.path.isdir(img_dir), f"Directory not found: {img_dir}"
        
        self.coco_data = COCO(annotations_file)
        self.subset_classes = subset_classes
        self.cluster_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.subset_classes.keys())}
        self.idx_to_cluster_class = {idx: cls_name for idx, cls_name in enumerate(self.cluster_class_to_idx.keys())}
        print(f"Cluster class to idx: {self.cluster_class_to_idx}")
        print(f"Idx to cluster class: {self.idx_to_cluster_class}")
        # Create reverse mapping
        self.reverse_mapping = {
            original_id: self.cluster_class_to_idx[cls_name]
            for cls_name, original_ids in self.subset_classes.items()
            for original_id in original_ids
        }
        print(f"Reverse mapping: {self.reverse_mapping}")
        
        
        # Get all annotation IDs
        all_ann_ids = self.coco_data.getAnnIds()
        
        # Filter annotations based on subset classes
        self.anns_ids = [
            ann_id for ann_id in all_ann_ids
            if self.coco_data.loadAnns(ann_id)[0]['category_id'] in self.reverse_mapping
        ]
        
        # Get unique image IDs that have annotations in our subset
        self.img_ids = list(set(
            self.coco_data.loadAnns(ann_id)[0]['image_id']
            for ann_id in self.anns_ids
        ))
        
        self.img_dir = img_dir
        self.transforms = transforms


    def __len__(self) -> None:
        """ Returns the length of the dataset """
        return len(self.img_ids)
    
    
    def square_img(self, img, bbox):
        """Make the image square by padding to at least 224x224"""
        MIN_SIZE = 224
        x_min, y_min, width, height = [int(x) for x in bbox]
        
        # Ensure width and height are positive
        width = max(1, width)
        height = max(1, height)
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        
        # Ensure bbox coordinates are within image bounds
        x_min = max(0, min(x_min, img_width - width))
        y_min = max(0, min(y_min, img_height - height))
        
        # Crop image to the bounding box
        cropped = img[y_min:y_min+height, x_min:x_min+width]
        
        # Calculate required padding to make it square and at least MIN_SIZE
        side_length = max(width, height, MIN_SIZE)
        
        # Calculate padding for square shape
        pad_left = (side_length - width) // 2
        pad_right = side_length - width - pad_left
        pad_top = (side_length - height) // 2
        pad_bottom = side_length - height - pad_top
        
        # Add padding
        squared_img = cv2.copyMakeBorder(
            cropped,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        return squared_img


    def __getitem__(self, idx) -> dict:
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
        category_id = self.reverse_mapping[annotation['category_id']]
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

        # Avoid issues of rotated images by rotating it accoring to EXIF info
        sample_img = ImageOps.exif_transpose(sample_img)

        # Generate the mask from the annotation segmentation
        mask = self.coco_data.annToMask(annotation)
        # print(f"Mask shape: {mask.shape}")

        # Make the image a numpy array to apply the mask
        sample_img = np.array(sample_img)
        # print(f"Sample image shape: {sample_img.shape}")

        # Apply the mask to the image
        cropped_image = cv2.bitwise_and(sample_img, sample_img, mask=mask)

        # Make square
        squared_img = self.square_img(cropped_image, bbox)
        
        # Convert to PIL Image for transforms
        # sample_img = Image.fromarray(sample_img)
        squared_img = np.array(squared_img)
        
        # Apply transforms
        if self.transforms:
          #   print("Applying transforms")
            transformed_img = self.transforms(squared_img)
        else:
            transformed_img = squared_img

        # print(f"Sample image shape: {squared_img.shape}")
        # print(f"Sample image type: {squared_img.dtype}")
        # print(f"Mask shape: {mask.shape}")
        # print(f"Mask type: {mask.dtype}")


        ################################################
        # # Display the image and the mask
        # # Convert tensor to numpy array and transpose dimensions for visualization
        # if isinstance(sample_img, torch.Tensor):
        #     vis_img = sample_img.permute(1, 2, 0).numpy()  # Change from (C,H,W) to (H,W,C)
        # else:
        #     vis_img = sample_img
            
        # plt.figure(figsize=(18, 5))
        # plt.subplot(1, 5, 1)
        # plt.imshow(vis_img)
        # # Set plot title equal to category name and image id
        # category_name = self.idx_to_cluster_class[category_id]
        # plt.title(f'{category_name} - Image ID: {img_id}')
        # plt.subplot(1, 5, 2)
        # plt.imshow(mask, cmap='gray')
        # plt.title('Mask')
        # plt.subplot(1, 5, 3)
        # if isinstance(cropped_image, torch.Tensor):
        #     cropped_vis = cropped_image.permute(1, 2, 0).numpy()
        # else:
        #     cropped_vis = cropped_image
        # plt.imshow(cropped_vis)
        # plt.title('Masked Image')
        # plt.subplot(1, 5, 4)
        # if isinstance(squared_img, torch.Tensor):
        #     squared_vis = squared_img.permute(1, 2, 0).numpy()
        # else:
        #     squared_vis = squared_img
        # plt.imshow(squared_vis)
        # plt.title('Squared Image')
        # plt.subplot(1, 5, 5)
        # if isinstance(transformed_img, torch.Tensor):
        #     transformed_vis = transformed_img.permute(1, 2, 0).numpy()
        # else:
        #     transformed_vis = transformed_img
        # plt.imshow(transformed_vis)
        # plt.title('Transformed Image')
        # plt.show()

        return {
            'pixel_values': transformed_img,
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