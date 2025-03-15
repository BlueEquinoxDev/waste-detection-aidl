import torch
from torch.utils.data import DataLoader
from transformers import MaskFormerImageProcessor, AutoImageProcessor, MaskFormerConfig, MaskFormerForInstanceSegmentation, Mask2FormerForUniversalSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from custom_datasets.taco_dataset_mask2former import TacoDatasetMask2Former, visualize_sample, visualize_batch
from utilities.maskformer_display_sample_results import display_sample_results
from torchmetrics.segmentation import MeanIoU
from PIL import Image, ImageOps

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update the hyperparameters
h_params = {
    "batch_size": 1,  # Force batch size to 1
    "num_workers": 0,
    "model_name": "facebook/mask2former-swin-tiny-ade-semantic",
    "checkpoint_path": "app/checkpoint/checkpoint_epoch_18_mask2former_taco1.pt",
    "dataset_name": "taco1", 
}

# Initialize processor with proper configuration
processor = MaskFormerImageProcessor.from_pretrained(h_params["model_name"])

# Create transform pipeline for consistent preprocessing
data_transforms_test = A.Compose([
    A.Resize(height=512, width=512),
    ToTensorV2()
])

# Initialize dataset with transforms
test_taco_dataset = TacoDatasetMask2Former(
    annotations_file="data/test_annotations.json",
    img_dir="data/images",
    processor=processor,
    transforms=data_transforms_test
)

idx2class = test_taco_dataset.idx2class

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"].float() / 255.0 for example in batch])  # Convert to float and normalize
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    original_images = [example["original_image"] for example in batch]
    original_masks = [example["original_mask"] for example in batch]
    image_ids = [example["image_id"] for example in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
        "original_images": original_images,
        "original_masks": original_masks,
        "image_id": image_ids
    }

test_loader = DataLoader(test_taco_dataset, 
                              batch_size = h_params["batch_size"],
                              num_workers = h_params["num_workers"],
                              shuffle=False,
                              collate_fn=collate_fn,
                              pin_memory=True)

checkpoint = torch.load(h_params["checkpoint_path"], map_location=device)

# Load model with configuration
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    h_params["model_name"],
    num_labels=len(idx2class),
    #config=model_config,
    ignore_mismatched_sizes=True
)

# Update model's state dict
try:
    model.load_state_dict(checkpoint['model_state_dict'])  # , strict=False
except RuntimeError as e:
    print(f"Warning: Some weights could not be loaded: {e}")

model.to(device)

metric = MeanIoU(num_classes=len(idx2class)+1, per_class=True, include_background=True, input_format="index")

results_masks=[]
images_not_predicts=[]
def test_one_epoch():      
    pbar = tqdm(test_loader, desc="Computing metrics test dataset", leave=False)
    model.eval()
    for batch_idx, batch in enumerate(pbar):
        # Since batch size is 1, we can directly access the first (and only) item
        pixel_values = batch["pixel_values"].to(device)  # Shape: [1, C, H, W]
        #print(f"pixel_values.shape: {pixel_values.shape}")
        image_id = batch["image_id"]  # Get the single image ID
        #print(f"Processing image ID: {image_id}")
        
        target_masks = torch.stack(batch["mask_labels"])
        target_masks_reshaped_all_labels = []
        #print(batch["class_labels"])
        #print(np.unique(batch["class_labels"]))
        for label in np.unique(batch["class_labels"]):
            if label == 0:
                pass
            else:
                #print(f"label: {label}")
                target_masks_reshaped = torch.zeros(target_masks.shape[-2:])

                #print("filter")
                label_filter = [batch["class_labels"][0] == label][0]
                filtered_masks = target_masks.squeeze()[label_filter]
                #print(filtered_masks)
                target_masks_reshaped = filtered_masks.sum(dim=0)
                #print(target_masks_reshaped)
                    
                #print(target_masks_reshaped.shape)
                #print(np.unique(target_masks_reshaped))
                target_masks_reshaped_all_labels.append(target_masks_reshaped)
        targets = torch.stack(target_masks_reshaped_all_labels).unsqueeze(0)
        # print(f"Target_shape: {targets.shape}")


        
        with torch.no_grad():                        
            outputs = model(pixel_values)

            target_sizes = [(mask.shape[1], mask.shape[0]) for mask in batch['original_masks']]
            # Compute predictions
            pred_maps = processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes, threshold=0.5, return_binary_maps=True
            )
            #print(f"raw_pred_maps: {pred_maps}")

            # Invert background and class
            predictions = 1 - torch.stack([pred["segmentation"] for pred in pred_maps])
            #print(predictions)
            #print(np.unique(predictions))
            
            #print(f"Predictions_shape: {predictions.shape}")
            
            """mask = predictions.squeeze(0).squeeze(0)
            mask_img = Image.fromarray(mask.numpy() *100)
            mask_img.show()"""
            
            #batch["original_images"][0].show()

            #print(metric(preds=predictions.type(torch.long), target=targets.type(torch.long)))

            metric.update(preds=predictions.type(torch.long), target=targets.type(torch.long))
            #display_sample_results(batch, outputs, processor, sample_index=0, mask_threshold=0.35, checkpoint_path=h_params["checkpoint_path"])
        

    
test_one_epoch()
results = metric.compute()
print(results)

idx2class[0] = "backgroud"

print("METRICS:")
for i, metric in enumerate(results):
    print(f"{idx2class[i]}: {metric}")

