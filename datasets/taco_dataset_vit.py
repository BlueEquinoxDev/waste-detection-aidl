from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image
#from utilities.config_utils import TaskType, ClassificationCategoryType
#from utilities.get_supercategory_by_id import get_supercategory_by_id
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
        category_id = annotation['category_id']
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
            mask = np.rot90(mask)

        # Apply the mask to the image
        cropped_image = cv2.bitwise_and(sample_img, sample_img, mask=mask)

        # Make square
        x_min, y_min, width, height = [int(dim) for dim in bbox]
        sample_img = self.square_img(width, height, cropped_image)
        
        # Convert to PIL Image for transforms
        # sample_img = Image.fromarray(sample_img)
        sample_img = np.array(sample_img)
        
        # Apply transforms
        if self.transforms:
          #   print("Applying transforms")
            sample_img = self.transforms(sample_img)

        # print(f"Sample image shape: {sample_img.shape}")
        # print(f"Sample image type: {sample_img.dtype}")
        # print(f"Mask shape: {mask.shape}")
        # print(f"Mask type: {mask.dtype}")
        # plt.imshow(sample_img)
        # plt.show()

        return {
            'pixel_values': sample_img,
            'labels': category_id
        }

        
# TESTING THE DATASET
# taco_dataset = TacoDatasetViT(annotations_file='data/train_annotations.json', img_dir='data')
# print(taco_dataset[22])
    