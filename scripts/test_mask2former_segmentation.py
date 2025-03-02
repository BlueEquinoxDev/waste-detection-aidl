import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, MaskFormerConfig, MaskFormerForInstanceSegmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from custom_datasets.taco_dataset_mask2former import TacoDatasetMask2Former, visualize_sample

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update the hyperparameters
h_params = {
    "batch_size": 1,  # Force batch size to 1
    "num_workers": 0,
}

# Initialize processor with proper configuration
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-base-coco-instance",
    do_resize=True,
    do_rescale=False,  # Disable rescaling since we handle it in transforms
    do_normalize=False  # Disable normalization since we handle it in transforms
)

# Create transform pipeline for consistent preprocessing
data_transforms_test = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
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

# Update collate_fn to not normalize again
def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])  # Don't normalize here
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    # Add image_id to the batch
    image_ids = [example["image_id"] for example in batch]
    
    return {
        "pixel_values": pixel_values, 
        "pixel_mask": pixel_mask, 
        "class_labels": class_labels, 
        "mask_labels": mask_labels,
        "image_id": image_ids
    }

test_loader = DataLoader(test_taco_dataset, 
                              batch_size = h_params["batch_size"],
                              num_workers = h_params["num_workers"],
                              shuffle=False,
                              collate_fn=collate_fn,
                              pin_memory=True)

checkpoint_path = "results/seg-mask2former-taco-original-backbone_frezze-augmentation-20250227-120453/best_model.pth/checkpoint_epoch_1_2025_2_27_12_21.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

model_config = MaskFormerConfig.from_pretrained(
    "facebook/mask2former-swin-base-coco-instance",
    num_labels=len(idx2class),
    output_hidden_states=True,
    output_attentions=True,
    use_auxiliary_loss=True,
    id2label=idx2class
)

# Load model with configuration
model = MaskFormerForInstanceSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-coco-instance",
    config=model_config,
    ignore_mismatched_sizes=True
)

# Update model's state dict
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
except RuntimeError as e:
    print(f"Warning: Some weights could not be loaded: {e}")
model.to(device)

def coco_result_format(image_id: str, prediction: dict, threshold: float = 0.5) -> None:
    # Get the first item from batch for each tensor
    labels = prediction['labels'][0]  # Shape: [num_queries]
    masks = prediction['masks'][0]    # Shape: [num_queries, H, W]
    scores = prediction['scores'][0]  # Shape: [num_queries]

    def mask_to_coco_format(mask):
        a=mask.cpu().squeeze().numpy()
        a=(a>=threshold).astype(np.uint8)
        a=np.asfortranarray(a)
        return encode(a)
    
    # Process each prediction that meets confidence threshold
    for i in range(len(scores)):
        if scores[i].item() > threshold:
            # Format mask result
            results_masks.append({
                "image_id": image_id,
                "category_id": labels[i].item(),
                "segmentation": mask_to_coco_format(masks[i]),
                "score": scores[i].item()
            })
    return results_masks

results_masks=[]
images_not_predicts=[]
def validation_one_epoch():      
    pbar = tqdm(test_loader, desc="Computing metrics test dataset", leave=False)
    model.eval()
    for batch_idx, batch in enumerate(pbar):
        # Since batch size is 1, we can directly access the first (and only) item
        pixel_values = batch["pixel_values"].to(device)  # Shape: [1, C, H, W]
        print(f"pixel_values.shape: {pixel_values.shape}")
        image_id = batch["image_id"][0]  # Get the single image ID
        print(f"Processing image ID: {image_id}")
        
        visualize_sample(batch)
        
        with torch.no_grad():                        
            outputs = model(pixel_values)
                     
            # Post-process predictions - remove batch dimension
            pred_masks = outputs.masks_queries_logits.sigmoid().squeeze(0)  # Remove batch dim
            pred_class_logits = outputs.class_queries_logits.squeeze(0)  # Remove batch dim
            
            # Get predictions - without batch dimension
            predictions = {
                'labels': pred_class_logits.argmax(dim=-1),      # [num_queries]
                'masks': pred_masks > 0.5,                       # [num_queries, H, W]
                'scores': pred_class_logits.softmax(dim=-1).max(dim=-1)[0],  # [num_queries]
            }

            print(f"Predictions for image ID: {image_id}")
            print(predictions)

            # # --- Debug: Display image with predicted mask ---
            # # Convert image tensor from CxHxW to HxWxC and bring it to CPU
            # img = pixel_values.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # # Choose the mask with the highest score
            # scores = predictions['scores']
            # max_score_idx = scores.argmax().item()
            # mask = predictions['masks'][max_score_idx].cpu().numpy()
            
            # plt.figure(figsize=(8, 8))
            # plt.imshow(img)
            # plt.imshow(mask, alpha=0.1, cmap='jet')
            # plt.title(f"Image ID: {image_id} - Predicted Mask (Score: {scores[max_score_idx].item():.2f})")
            # plt.axis('off')
            # plt.show()
            
            # Add batch dimension back for COCO evaluation format
            predictions = {
                'labels': pred_class_logits.argmax(dim=-1).unsqueeze(0),
                'masks': (pred_masks > 0.5).unsqueeze(0),
                'scores': pred_class_logits.softmax(dim=-1).max(dim=-1)[0].unsqueeze(0)
            }
            
            # Format results for COCO evaluation
            coco_result_format(image_id, predictions)
        
    print("COCO metrics for masks:\n")            
    coco_result = test_loader.dataset.coco_data.loadRes(results_masks)
    coco_eval = COCOeval(test_loader.dataset.coco_data, coco_result, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
# validation_one_epoch()

sample = test_taco_dataset[2]
img = sample["pixel_values"]
print("Shape:", img.shape)
print("Min:", img.min(), "Max:", img.max(), "Mean:", img.mean())

# Visualize the sample
visualize_sample(sample)
