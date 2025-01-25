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
        
        # Get the clustercategory mapping
        with open('data/clustercategories.json', 'r') as infile:
            clustercategories_list = json.load(infile)
        
        # Mapping from supercategory to cluster category
        self.supercategory_to_cluster = {}

        for cluster in clustercategories_list:
            cluster_id = cluster["id"]
            for supercategory_id in cluster["supercategories"]:
                self.supercategory_to_cluster[supercategory_id] = cluster_id

        print(f"self.supercategory_to_cluster: {self.supercategory_to_cluster}")
        
        # create mappings from the list of cluster categories
        self.idx_to_cluster_class = {item['id']: item['cluster-category']
                            for item in clustercategories_list}
        self.cluster_class_to_idx = {item['cluster-category']: item['id']
                            for item in clustercategories_list}


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
        category_id = self.supercategory_to_cluster[self.category_map[annotation['category_id']]]
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
        # if the mask is not the same size as the image, rotate it
        if mask.shape != sample_img.shape[:2]:
            # display the image and the mask
            # plt.figure(figsize=(20, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(sample_img)
            # plt.title('Original Image')
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask, cmap='gray')
            # plt.title('Mask')
            # plt.show()
            # rotate the mask 270 degrees
            mask = np.rot90(mask, 3)

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