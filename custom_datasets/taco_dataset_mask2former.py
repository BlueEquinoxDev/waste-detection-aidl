from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image, ImageOps
import numpy as np
import torch
import json
from torchvision import tv_tensors
import matplotlib.pyplot as plt

class TacoDatasetMask2Former(Dataset):

    def __init__(self,
                 annotations_file: str, 
                 img_dir: str,
                 processor,
                 transforms=None) -> None:
        """ Constructor for the TacoDataset class """
        super().__init__()

        assert os.path.isfile(annotations_file), f"File not found: {annotations_file}"
        assert os.path.isdir(img_dir), f"Directory not found: {img_dir}"
        
        self.coco_data = COCO(annotations_file)
        self.len_dataset=len(self.coco_data.imgs.keys())
        self.img_dir = img_dir
        self.processor = processor
        self.transforms = transforms
        
        self.index_to_imageId={i:img_id for i,img_id in enumerate(self.coco_data.imgs.keys())}
        
        # Get categories from COCO data instead of external JSON
        categories = self.coco_data.loadCats(self.coco_data.getCatIds())
        
        # Get unique supercategories
        supercategories = list(set(cat['supercategory'] for cat in categories))
        
        # Create mappings (add 1 to indices for background class at 0)
        self.idx2class = {i+1: supercategory for i, supercategory in enumerate(supercategories)}
        self.idx2class[0] = "background"  # Add background class
        
        self.class2idx = {supercategory: i+1 for i, supercategory in enumerate(supercategories)}
        self.class2idx["background"] = 0
        
        # Create category mapping for COCO annotations
        self.category_map = {cat['id']: self.class2idx[cat['supercategory']] 
                            for cat in categories}
        
    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, idx):

        img_id=self.index_to_imageId[idx]                        
        img_coco_data = self.coco_data.loadImgs([img_id])[0]
        path = os.path.join(self.img_dir, img_coco_data['file_name'])
        annotations = self.coco_data.imgToAnns[img_id]

        # Load and process image        
        sample_img = Image.open(path)
        sample_img = ImageOps.exif_transpose(sample_img)
        sample_img = np.array(sample_img)
        
        # Get masks and labels for each annotation
        masks =[]
        labels=[]
        for ann in annotations:
            masks.append(self.coco_data.annToMask(ann))
            labels.append(self.category_map[ann['category_id']])

        # Stack masks and transpose to (N, H, W) format
        instance_seg = np.stack(masks) if masks else np.zeros((0, sample_img.shape[0], sample_img.shape[1]))
        class_labels = torch.tensor(labels, dtype=torch.int64)
        inst2class = {0: 0}  # Background mapping: 0 -> 0
        for i, label in enumerate(labels):
            inst2class[i + 1] = label

        # print(f"sample_img.shape: {sample_img.shape}")
        # print(f"sample_img: {sample_img}")
        # print(f"instance_seg.shape: {instance_seg.shape}")
        # print(f"class_labels: {class_labels}")
        # print(f"inst2class: {inst2class}")

        # Apply transforms to both image and mask
        if self.transforms is not None:
            # Transform image
            transformed = self.transforms(image=sample_img)
            # print(f"transformed: {transformed}")
            image = transformed['image']  # Now in torch.Tensor format (C,H,W)
            # Convert image to (H, W, C) format
            # image = image.permute(1, 2, 0)
            # print(f"image.shape: {image.shape}")
            
            # Transform masks
            if len(masks) > 0:
                transformed_masks = []
                for mask in instance_seg:
                     # Add channel dimension for Albumentations
                    mask = np.expand_dims(mask, axis=-1)
                    mask_transformed = self.transforms(image=mask)['image']
                    transformed_masks.append(mask_transformed)
                instance_seg = torch.stack(transformed_masks)  # [N, C, H, W]
                # Remove the extra channel dimension
                instance_seg = instance_seg.squeeze(1)  # [N, H, W]
            else:
                instance_seg = torch.zeros((0, 512, 512))

        if class_labels.shape[0] == 0:
            # print("No annotations found")
            inputs = self.processor(
                images=[image.numpy()],
                return_tensors="pt"
            )
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, 512, 512))
        else:
        # Convert instance_seg to numpy of type uint8
            instance_seg = instance_seg.numpy().astype(np.uint8)
            # print(f"instance_seg.shape: {instance_seg.shape}")
            # print(f"np.unique(instance_seg): {np.unique(instance_seg)}")
            H, W = instance_seg.shape[1], instance_seg.shape[2]

            # Create a semantic segmentation map with background=0
            semantic_seg = np.zeros((H, W), dtype=np.uint8)
            for i, mask in enumerate(instance_seg):
                # Assign a unique instance ID (i+1) to the mask region
                # print(f"i: {i}, mask.shape: {mask.shape}")
                semantic_seg[mask > 0] = i + 1

            # Add a channel dimension (if required by the processor)
            # semantic_seg = np.expand_dims(semantic_seg, axis=-1)  # shape: (H, W, 1)

            # Convert semantic_seg to numpy of type uint8
            semantic_seg = semantic_seg.astype(np.uint8)

            # Ensure the segmentation map is properly padded to match expected input size
            print(f"semantic_seg.shape: {semantic_seg.shape}")
            print(f"np.unique(semantic_seg): {np.unique(semantic_seg)}")
            print(f"semantic_seg.dtype: {semantic_seg.dtype}")
            print(f"inst2class: {inst2class}")
            # target_size = (512, 512, 1)
            # padded_segmentation = np.zeros(target_size, dtype=np.uint8)
            
            # # padded_segmentation[:H, :W] = semantic_seg[:, :, 0]  # Remove extra channel

            # print(f"padded_segmentation.shape: {padded_segmentation.shape}")
            # print(f"np.unique(padded_segmentation): {np.unique(padded_segmentation)}")

            # Process using the MaskFormer processor
            inputs = self.processor(
                images=[image.numpy()], 
                segmentation_maps=[semantic_seg],  
                instance_id_to_semantic_id=inst2class,
                # segmentation_pad_value=0,
                # padding=255,
                return_tensors="pt"
            )
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

            ############## DEBUGING #############
            visualize_sample(inputs)
            ######################################
        return inputs

        # Output Format
        # pixel_values torch.Size([3, 512, 512])
        # pixel_mask torch.Size([512, 512])
        # class_labels torch.Size([1])
        # mask_labels torch.Size([0, 512, 512])

def visualize_sample(inputs):
    """
    Visualize a sample from the dataset.
    This function takes a dictionary of inputs containing pixel values, pixel masks, 
    mask labels, and class labels, and visualizes the original image alongside 
    the segmentation masks.
    Args:
        inputs (dict): A dictionary containing the following keys:
            - 'pixel_values' (torch.Tensor): A tensor representing the pixel values of the image.
            - 'pixel_mask' (torch.Tensor): A tensor representing the pixel mask.
            - 'mask_labels' (torch.Tensor): A tensor containing the segmentation masks.
            - 'class_labels' (list): A list of class labels corresponding to the masks.
    Returns:
        None: This function displays the images using matplotlib and does not return any value.
    """
    # Convert tensor to numpy array and transpose to (H, W, C) format
    pixel_values = inputs['pixel_values'].numpy().transpose(1, 2, 0)
    pixel_mask = inputs['pixel_mask'].numpy()
    mask_labels = inputs['mask_labels'].numpy()
    class_labels = inputs['class_labels'].tolist()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axes[0].imshow(pixel_values)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display image with superposed masks
    axes[1].imshow(pixel_values)  # Base image
    
    if mask_labels.size > 0:
        # Generate distinct colors for each mask
        num_masks = mask_labels.shape[0]
        colors = plt.cm.rainbow(np.linspace(0, 1, num_masks))
        
        # Superpose each mask with a different color and 50% transparency
        for i, mask in enumerate(mask_labels):
            # Create masked array for better visualization
            masked = np.ma.masked_where(mask == 0, mask)
            axes[1].imshow(masked, cmap=plt.cm.colors.ListedColormap([colors[i]]), 
                         alpha=0.5)  # 50% transparency
    
    axes[1].set_title(f"Segmentation Masks\nClass Labels: {class_labels}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


