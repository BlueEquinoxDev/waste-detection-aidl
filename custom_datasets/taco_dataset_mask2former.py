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
        img_id = self.index_to_imageId[idx]                        
        img_coco_data = self.coco_data.loadImgs([img_id])[0]
        path = os.path.join(self.img_dir, img_coco_data['file_name'])
        annotations = self.coco_data.imgToAnns[img_id]

        # Load and process image        
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        image = np.array(image)
        
        # Get masks and labels for each annotation
        masks = []
        labels = []
        for ann in annotations:
            masks.append(self.coco_data.annToMask(ann))
            labels.append(self.category_map[ann['category_id']])

        # Stack masks and transpose to (N, H, W) format
        instance_seg = np.stack(masks) if masks else np.zeros((0, image.shape[0], image.shape[1]))
        class_labels = torch.tensor(labels, dtype=torch.int64)
        inst2class = {0: 0}  # Background mapping: 0 -> 0
        for i, label in enumerate(labels):
            inst2class[i + 1] = label

        # Apply transforms to both image and mask
        if self.transforms is not None:
            if len(masks) > 0:
                transformed = self.transforms(image=image, masks=list(instance_seg))
                image = transformed['image']  # In range [0, 1] after normalization
                instance_seg = torch.stack([torch.tensor(m, dtype=torch.uint8) for m in transformed['masks']])
            else:
                transformed = self.transforms(image=image)
                image = transformed['image']
                instance_seg = torch.zeros((0, 512, 512))

        # Convert to numpy array in range [0, 1]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            
        # Rescale image to [0, 1] range if needed
        if image.max() > 1.0 or image.min() < 0.0:
            image = (image - image.min()) / (image.max() - image.min())

        if class_labels.shape[0] == 0:
            inputs = self.processor(
                images=[image],
                return_tensors="pt"
            )
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, 512, 512))
        else:
            # Convert instance_seg to semantic_seg
            semantic_seg = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
            for i, mask in enumerate(instance_seg):
                semantic_seg[mask > 0] = i + 1

            inputs = self.processor(
                images=[image],
                segmentation_maps=[semantic_seg],
                instance_id_to_semantic_id=inst2class,
                return_tensors="pt"
            )
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        inputs["image_id"] = img_id
        # visualize_sample(inputs)
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
            - 'pixel_values' (torch.Tensor): A tensor representing the pixel values of the image. Size is (C, H, W).
            - 'pixel_mask' (torch.Tensor): A tensor representing the pixel mask. Size is (H, W).
            - 'mask_labels' (torch.Tensor): A tensor containing the segmentation masks. Size is (N, H, W).
            - 'class_labels' (list): A list of class labels corresponding to the masks. Size is (N).
            - 'image_id' (int): The image ID.
    Returns:
        None: This function displays the images using matplotlib and does not return any value.
    """
    # print(f"Input Format\n{inputs}")

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
        
        # Superpose each mask with a different color and 70% transparency
        for i, mask in enumerate(mask_labels):
            if i != 0:
                # Create masked array for better visualization
                masked = np.ma.masked_where(mask == 0, mask)
                axes[1].imshow(masked, cmap=plt.cm.colors.ListedColormap([colors[i]]), 
                            alpha=0.7)  # 70% transparency
    
    axes[1].set_title(f"Segmentation Masks\nClass Labels: {class_labels}")
    axes[1].axis('off')

    plt.savefig('test_image.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


