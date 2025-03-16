from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image, ImageOps
import numpy as np
import torch
import json
from torchvision import tv_tensors
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

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
        #supercategories = list(set(cat['supercategory'] for cat in categories))
        
        self.idx2class= {self.coco_data.cats[i]['id']: self.coco_data.cats[i]['supercategory'] for i in self.coco_data.cats}
        self.idx2class=dict(sorted(self.idx2class.items()))
        
        # Create mappings (add 1 to indices for background class at 0)
        #self.idx2class = {i+1: supercategory for i, supercategory in enumerate(supercategories)}
        #self.idx2class[0] = "background"  # Add background class
        
        #self.class2idx = {supercategory: i+1 for i, supercategory in enumerate(supercategories)}
        #self.class2idx["background"] = 0
        
        # Create category mapping for COCO annotations
        #self.category_map = {cat['id']: self.class2idx[cat['supercategory']] 
        #                    for cat in categories}
        
    def __len__(self):
        return self.len_dataset
    
    def __getitem__(self, idx):
        img_id = self.index_to_imageId[idx]                        
        img_coco_data = self.coco_data.loadImgs([img_id])[0]
        path = os.path.join(self.img_dir, img_coco_data['file_name'])
        annotations = self.coco_data.imgToAnns[img_id]

        # Load and process image        
        sample_img = Image.open(path)
        sample_img = ImageOps.exif_transpose(sample_img)
        
        orginal_img = sample_img.copy()

        sample_img = np.array(sample_img)
        
        # Get masks and labels for each annotation
        masks = []
        labels = []
        for ann in annotations:
            masks.append(self.coco_data.annToMask(ann))
            labels.append(ann['category_id'])

        #print(f"labels: {labels}")
        #print(f"labels to text: {[self.idx2class[i] for i in labels]}")

        # Stack masks and transpose to (N, H, W) format
        instance_seg = np.stack(masks) if masks else np.zeros((0, image.shape[0], image.shape[1]))
        class_labels = torch.tensor(labels, dtype=torch.int64)
        inst2class = {0: 0}  # Background mapping: 0 -> 0
        for i, label in enumerate(labels):
            inst2class[i + 1] = label
        
        # print(f"sample_img.shape: {sample_img.shape}")
        # print(f"sample_img: {sample_img}")
        #print(f"instance_seg.shape: {instance_seg.shape}")
        # print(f"class_labels: {class_labels}")
        #print(f"inst2class: {inst2class}")

        # Apply transforms to both image and mask
        if self.transforms is not None:
            if len(masks) > 0:
                transformed = self.transforms(image=sample_img, masks=list(instance_seg))
                image = transformed['image']
                instance_seg = torch.stack([m.type(torch.uint8) for m in transformed['masks']])
            else:
                transformed = self.transforms(image=image)
                image = transformed['image']
                instance_seg = torch.zeros((0, 512, 512))
            
            #print(f"instance_seg.shape after tansforms: {instance_seg.shape}")

            # # Transform image
            # transformed = self.transforms(image=sample_img)
            # # print(f"transformed: {transformed}")
            # image = transformed['image']  # Now in torch.Tensor format (C,H,W)
            # # Convert image to (H, W, C) format
            # # image = image.permute(1, 2, 0)
            # # print(f"image.shape: {image.shape}")
            
            # # Transform masks
            # if len(masks) > 0:
            #     transformed_masks = []
            #     for mask in instance_seg:
            #          # Add channel dimension for Albumentations
            #         mask = np.expand_dims(mask, axis=-1)
            #         mask_transformed = self.transforms(image=mask)['image']
            #         transformed_masks.append(mask_transformed)
            #     instance_seg = torch.stack(transformed_masks)  # [N, C, H, W]
            #     # Remove the extra channel dimension
            #     instance_seg = instance_seg.squeeze(1)  # [N, H, W]
            # else:
            #     instance_seg = torch.zeros((0, 512, 512))

        if class_labels.shape[0] == 0:
            print("No annotations found")
            inputs = self.processor(
                images=[image],
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
            #print(f"semantic_seg.shape: {semantic_seg.shape}")
            #print(f"semantic_seg unique: {np.unique(semantic_seg)}")
            #print(f"instance2class: {inst2class}")
            #print(f"labels: {[self.idx2class[i] for i in inst2class.values()]}")

            original_mask = semantic_seg.copy()

            # Ensure the segmentation map is properly padded to match expected input size
            # print(f"semantic_seg.shape: {semantic_seg.shape}")
            # print(f"np.unique(semantic_seg): {np.unique(semantic_seg)}")
            # print(f"semantic_seg.dtype: {semantic_seg.dtype}")
            # print(f"inst2class: {inst2class}")
            # target_size = (512, 512, 1)
            # padded_segmentation = np.zeros(target_size, dtype=np.uint8)
            
            # # padded_segmentation[:H, :W] = semantic_seg[:, :, 0]  # Remove extra channel

            # print(f"padded_segmentation.shape: {padded_segmentation.shape}")
            # print(f"np.unique(padded_segmentation): {np.unique(padded_segmentation)}")

            # Process using the MaskFormer processor
            inputs = self.processor(
                images=image.numpy(), 
                segmentation_maps=semantic_seg,  
                instance_id_to_semantic_id=inst2class,
                return_tensors="pt",
                do_resize=False,
                do_rescale=False,
                do_normalize=False
            )
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
            inputs["original_image"] = orginal_img
            inputs["original_mask"] = original_mask
            inputs["image_id"] = img_id
            inputs["inst2class"] = inst2class

            #print(inputs)
            
            ############## DEBUGING #############
            #visualize_sample(inputs, idx2class=self.idx2class)
            ######################################
        return inputs

        # Output Format
        # pixel_values torch.Size([3, 512, 512])
        # pixel_mask torch.Size([512, 512])
        # class_labels torch.Size([1])
        # mask_labels torch.Size([0, 512, 512])

def visualize_batch(batch, idx2class, results_masks=None):

    batch_size = batch['pixel_values'].shape[0]

    for i in range(batch_size):
        inputs = {k: v[i] for k, v in batch.items()}
        if results_masks is not None:
            inputs['results_masks'] = results_masks[i]
        visualize_sample(inputs, idx2class)



def visualize_sample(inputs, idx2class):
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
    results_masks = inputs.get('results_masks', None)
    # Determine number of subplots (original, masks, results_masks if available)
    num_subplots = 3 if results_masks is not None else 2
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_subplots, figsize=(15, 6))

    # Display the original image
    axes[0].imshow(pixel_values)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display image with superposed masks
    axes[1].imshow(pixel_values)  # Base image
    print(f"mask_labels.shape: {mask_labels.shape}")
    print(f"Class labels: {class_labels}")
    print(f"Class labels names {[idx2class[i] for i in class_labels]}")
    print(f"All class labels: {idx2class}")
    
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

    # Display results masks if available
    if results_masks is not None:
        axes[2].imshow(pixel_values)  # Base image

        print(f"Results masks: {results_masks}")
        
        # Handle different types of results_masks
        if isinstance(results_masks, dict):
            # For COCO format results with RLE-encoded masks
            from pycocotools import mask as mask_util
            
            # Create a mask from the RLE encoding
            if 'segmentation' in results_masks:
                rle = results_masks['segmentation']
                binary_mask = mask_util.decode(rle)
                
                # Resize mask if needed (your image might be 512x512 but mask is 128x128)
                if binary_mask.shape != (pixel_values.shape[0], pixel_values.shape[1]):
                    from skimage.transform import resize
                    binary_mask = resize(binary_mask, (pixel_values.shape[0], pixel_values.shape[1]), 
                                        order=0, preserve_range=True).astype(np.uint8)
                
                # Display the mask
                masked_result = np.ma.masked_where(binary_mask == 0, binary_mask)
                axes[2].imshow(masked_result, cmap='jet', alpha=0.7)
                
                # Add category and score as text
                category_id = results_masks.get('category_id', 'Unknown')
                score = results_masks.get('score', 0)
                axes[2].text(10, 30, f"Cat: {category_id}, Score: {score:.2f}", 
                           color='white', backgroundcolor='black', fontsize=10)
            else:
                axes[2].text(pixel_values.shape[1]//2, pixel_values.shape[0]//2, 
                           "Invalid mask format", color='red', fontsize=12)
                
        elif isinstance(results_masks, list):
            num_result_masks = len(results_masks)
            result_colors = plt.cm.rainbow(np.linspace(0, 1, num_result_masks))
            
            for i, mask in enumerate(results_masks):
                masked_result = np.ma.masked_where(mask == 0, mask)
                axes[2].imshow(masked_result, cmap=plt.cm.colors.ListedColormap([result_colors[i]]), 
                            alpha=0.7)

    plt.savefig('test_image.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
