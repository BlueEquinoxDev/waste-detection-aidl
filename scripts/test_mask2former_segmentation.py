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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update the hyperparameters
h_params = {
    "batch_size": 1,  # Force batch size to 1
    "num_workers": 0,
    "model_name": "facebook/mask2former-swin-tiny-ade-semantic",
    "checkpoint_path": "results/seg-mask2former-swin-tiny-ade-semantic-taco1-True_backbone_freeze-True_augmentation-20250307-105643/best_mask2former_model.pth/checkpoint_epoch_1_2025_3_7_11_41.pt",
    # "checkpoint_path": "results/seg-mask2former-swin-tiny-ade-semantic-taco1-encoder_freeze-with_augmentation-20250305-121712/mask2former_30.pth/checkpoint_epoch_30_2025_3_5_16_59.pt",
    # "checkpoint_path": "results/seg-mask2former-swin-tiny-ade-semantic-taco5-encoder_freeze-with_augmentation-20250305-190831/best_mask2former_model.pth/checkpoint_epoch_19_2025_3_5_22_56.pt",
    # "checkpoint_path": "results/seg-mask2former-swin-tiny-ade-semantic-taco5-True_backbone_freeze-True_augmentation-20250307-233024/best_mask2former_model.pth/checkpoint_epoch_18_2025_3_7_23_50.pt",
    # "checkpoint_path": "results/seg-taco-mask2former-swin-tiny-ade-semantic-False_backbone_freeze-False_augmentation-20250304-105327/best_model.pth/checkpoint_epoch_1_2025_3_4_11_4.pt",
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

model_config = MaskFormerConfig.from_pretrained(
    h_params["model_name"],
    num_labels=len(idx2class),
    output_hidden_states=True,
    output_attentions=True,
    use_auxiliary_loss=True,
    id2label=idx2class
)

# Load model with configuration
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    h_params["model_name"],
    num_labels=len(idx2class),
    #config=model_config,
    ignore_mismatched_sizes=True
)

# Update model's state dict
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
except RuntimeError as e:
    print(f"Warning: Some weights could not be loaded: {e}")

model.to(device)

def coco_result_format(image_id: str, prediction: dict, threshold: float = 0.01) -> None:
    # Process each item in the batch
    batch_size = prediction['labels'].shape[0]
    
    for batch_idx in range(batch_size):
        # Get the current batch item
        labels = prediction['labels'][batch_idx]     # Shape: [num_queries]
        masks = prediction['masks'][batch_idx]       # Shape: [num_queries, H, W]
        scores = prediction['scores'][batch_idx]     # Shape: [num_queries]
        
        # Get the corresponding image_id for this batch item
        current_image_id = image_id[batch_idx]

        def mask_to_coco_format(mask):
            a = mask.cpu().squeeze().numpy()
            a = (a >= threshold).astype(np.uint8)
            a = np.asfortranarray(a)
            return encode(a)
        
        # Process each prediction that meets confidence threshold
        for i in range(len(scores)):
            if scores[i].item() > threshold:
                # Format mask result
                results_masks.append({
                    "image_id": current_image_id,
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
        image_id = batch["image_id"]  # Get the single image ID
        print(f"Processing image ID: {image_id}")
        
        # visualize_batch(batch, idx2class)
        
        with torch.no_grad():                        
            outputs = model(pixel_values)
                     
            # Post-process predictions - keep batch dimension
            pred_masks = outputs.masks_queries_logits.sigmoid()  # Keep batch dim
            pred_class_logits = outputs.class_queries_logits  # Keep batch dim
            
            # Get predictions - with batch dimension
            predictions = {
                'labels': pred_class_logits.argmax(dim=-1),      # [batch_size, num_queries]
                'masks': pred_masks > 0.5,                       # [batch_size, num_queries, H, W]
                'scores': pred_class_logits.softmax(dim=-1).max(dim=-1)[0],  # [batch_size, num_queries]
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
            
            # # Add batch dimension back for COCO evaluation format
            # predictions = {
            #     'labels': pred_class_logits.argmax(dim=-1).unsqueeze(0),
            #     'masks': (pred_masks > 0.5).unsqueeze(0),
            #     'scores': pred_class_logits.softmax(dim=-1).max(dim=-1)[0].unsqueeze(0)
            # }
            
            # # Format results for COCO evaluation
            # results_masks = coco_result_format(image_id, predictions)
            # visualize_batch(batch, idx2class, results_masks)

            display_sample_results(batch, outputs, processor, sample_index=0, mask_threshold=0.35, checkpoint_path=h_params["checkpoint_path"])
        
    print("COCO metrics for masks:\n")            
    coco_result = test_loader.dataset.coco_data.loadRes(results_masks)
    coco_eval = COCOeval(test_loader.dataset.coco_data, coco_result, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
validation_one_epoch()

# sample = test_taco_dataset[2]
# img = sample["pixel_values"]
# print("Shape:", img.shape)
# #print("Min:", img.min(), "Max:", img.max(), "Mean:", img.mean())

# # Visualize the sample
# visualize_sample(sample, idx2class)
