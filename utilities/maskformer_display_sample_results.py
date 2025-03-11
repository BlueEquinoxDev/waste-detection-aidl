import numpy as np
import torch
import matplotlib.pyplot as plt
import random

def apply_mask(image, mask, color, alpha=0.5):
    """
    Overlays a binary mask on an image with a given color and transparency.
    
    Args:
        image (np.ndarray): Image in [0, 1] of shape (H, W, 3).
        mask (np.ndarray): Binary mask of shape (H, W); nonzero indicates mask.
        color (list or tuple): RGB color values in [0, 1].
        alpha (float): Transparency factor.
    
    Returns:
        np.ndarray: Image with the mask overlaid.
    """
    for c in range(3):
        image[..., c] = np.where(mask > 0,
                                 image[..., c] * (1 - alpha) + alpha * color[c],
                                 image[..., c])
    return image

def display_sample_results(batch, outputs, processor, sample_index=0, mask_threshold=0.5, checkpoint_path = "NA"):
    """
    Displays the input image with overlays of ground truth and predicted instance masks/labels.
    
    The function uses the processor's post_process_instance_segmentation method, which returns
    a dictionary per image with:
      - "segmentation": Either a tensor of shape (num_instances, H, W) (if using binary maps)
                        or a 2D tensor (H, W) where each pixel is a segment id.
      - "segments_info": A list of dictionaries for each segment with keys "id", "label_id", "score", etc.
    
    This function adapts to the output by checking the dimensions of the segmentation.
    
    Args:
        batch (dict): Contains:
            - "pixel_values": Tensor of shape (B, C, H, W) with normalized pixel values.
            - "mask_labels": Tensor of shape (B, num_gt_masks, H, W) with ground truth masks.
            - "class_labels": Tensor or list of shape (B, num_gt_masks) with ground truth labels.
        outputs: Raw outputs from the model during training.
        processor: Processor object (e.g. Mask2FormerProcessor) with post_process_instance_segmentation.
        sample_index (int): Which sample from the batch to visualize.
        mask_threshold (float): Threshold for binarizing soft masks.
    """
    # Convert the input image from (C, H, W) to (H, W, C)
    image = batch["pixel_values"][sample_index].permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)
    height, width = image.shape[:2]
    target_sizes = [(mask.shape[1], mask.shape[0]) for mask in batch['original_masks']]

    # Use the processor to post-process the outputs.
    print(f"Image shape: {(height, width)}")
    results = processor.post_process_instance_segmentation(
        outputs, threshold=mask_threshold, target_sizes=target_sizes#target_sizes=[(height, width)]
    )
    
    instance_results = results[sample_index]
    segmentation = instance_results["segmentation"]
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Segmentation: {segmentation}")
    
    if segmentation is None:
        print("No segmentation found above threshold for sample", sample_index)
        return

    # Determine if segmentation is 2D or 3D.
    if segmentation.ndim == 2:
        # segmentation is a 2D map where each pixel is a segment id.
        # Create binary masks by comparing to each segment id.
        pred_masks = []
        print(f"Segments info: {instance_results['segments_info']}")
        for seg_info in instance_results["segments_info"]:
            seg_id = seg_info["id"]
            binary_mask = (segmentation == seg_id).cpu().numpy().astype(np.uint8)
            pred_masks.append(binary_mask)
        if len(pred_masks) > 0:
            pred_masks = np.stack(pred_masks, axis=0)
        else:
            pred_masks = None
    elif segmentation.ndim == 3:
        # segmentation is already a tensor of binary masks.
        pred_masks = segmentation
    else:
        raise ValueError("Unexpected segmentation shape:", segmentation.shape)
    
    # Extract predicted labels from segments_info.
    pred_labels = [f'{seg_info["label_id"]} ({seg_info["score"]:.2f})' for seg_info in instance_results["segments_info"]]

    # Process ground truth masks and labels.
    gt_masks = batch["mask_labels"][sample_index].cpu().numpy()
    if torch.is_tensor(batch["class_labels"][sample_index]):
        gt_labels = batch["class_labels"][sample_index].cpu().numpy()
    else:
        gt_labels = np.array(batch["class_labels"][sample_index])
    
    # Create copies for overlay visualization.
    # Convert image to grayscale
    grayscale = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    
    # Create 3-channel grayscale images for overlays
    gt_overlay = np.stack([grayscale, grayscale, grayscale], axis=-1)
    pred_overlay = np.stack([grayscale, grayscale, grayscale], axis=-1)
    
    # Overlay ground truth masks.
    gt_annotations = []
    for i in range(gt_masks.shape[0]):
        color = [random.random() for _ in range(3)]
        mask = gt_masks[i]
        gt_overlay = apply_mask(gt_overlay, mask, color, alpha=0.8)
        # np.where should now return two arrays if mask is 2D.
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
            gt_annotations.append((center_x, center_y, str(gt_labels[i])))
    
    # Overlay predicted masks.
    pred_annotations = []
    if pred_masks is not None:
        for i in range(pred_masks.shape[0]):
            color = [random.random() for _ in range(3)]
            mask = pred_masks[i]
            # If mask is soft, threshold it.
            if mask.dtype != np.uint8 and mask.max() <= 1.0:
                mask = (mask > mask_threshold).astype(np.uint8)
            pred_overlay = apply_mask(pred_overlay, mask, color, alpha=0.8)
            # Ensure mask is 2D.
            if mask.ndim != 2:
                mask = mask.squeeze()
            ys_xs = np.where(mask > 0)
            if len(ys_xs) == 2:
                ys, xs = ys_xs
                if len(xs) > 0 and len(ys) > 0:
                    center_x = int(np.mean(xs))
                    center_y = int(np.mean(ys))
                    pred_annotations.append((center_x, center_y, str(pred_labels[i])))
    else:
        print("No predicted masks available for sample", sample_index)
    
    # Plot the results.
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth Masks & Labels")
    for (x, y, label) in gt_annotations:
        axes[1].text(x, y, label, color="white", fontsize=12, weight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Predicted Masks & Labels")
    for (x, y, label) in pred_annotations:
        axes[2].text(x, y, label, color="white", fontsize=12, weight="bold")
    axes[2].axis("off")

    # Print the checkpoint_path in the bottom of the image
    fig.text(0.5, 0.05, checkpoint_path, ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

