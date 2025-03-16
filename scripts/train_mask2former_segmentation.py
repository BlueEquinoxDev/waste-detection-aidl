from torch.utils.data import DataLoader
from transformers import MaskFormerImageProcessor, MaskFormerConfig
import numpy as np
from custom_datasets.taco_dataset_mask2former import TacoDatasetMask2Former 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import MaskFormerForInstanceSegmentation, AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utilities.save_model import save_model
from utilities.maskformer_display_sample_results import display_sample_results
import os
import evaluate
from PIL import Image, ImageOps
# from skimage.transform import resize
import torchvision.transforms as T
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Train mask2former')
parser.add_argument('--batch_size', required=False, help='Batch size', type=int, default=1)
parser.add_argument('--dataset_type', required=True, help='Name of the trained dataset', type=str)
parser.add_argument('--checkpoint_path', required=False, help='Checkpoint path', type=str, default="")
args = parser.parse_args()

h_params ={
    "batch_size": args.batch_size,
    "num_workers": 0,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "model_name": "facebook/mask2former-swin-tiny-ade-semantic",
    "dataset_name": args.dataset_type,
    "backbone_freeze": True,
    "augmentation": True,
    "backup_best_model": True,
    "load_checkpoint": args.checkpoint_path != "",
    "checkpoint_path": args.checkpoint_path,
}

# experiment_name contains model name, backbone_freeze, augmentation
experiment_name = "seg-"+h_params["model_name"].split("/")[1]+"-"+h_params["dataset_name"]+"-"+str(h_params["backbone_freeze"])+"_backbone_freeze-"+str(h_params["augmentation"])+"_augmentation"
print("Experiment name: ", experiment_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logdir = os.path.join("logs", f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
results_dir = os.path.join("results", f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")


# Initialize Tensorboard Writer with the previous folder 'logdir'
writer=SummaryWriter(log_dir=logdir)

# processor = MaskFormerImageProcessor(
#     reduce_labels=True,
#     ignore_index=255,
#     do_resize=False,
#     do_rescale=False,
#     do_normalize=False
# )

## Initialize processor with proper configuration
# processor = AutoImageProcessor.from_pretrained(
#     h_params["model_name"],
#     do_resize=True,
#     do_rescale=False,  # Disable rescaling since we handle it in transforms
#     do_normalize=False  # Disable normalization since we handle it in transforms
# )
#processor = Mask2FormerImageProcessor(
#        ignore_index=255, reduce_labels=True
#    )
processor = MaskFormerImageProcessor.from_pretrained(h_params["model_name"])

# Create transform pipeline that handles both image and mask
if h_params["augmentation"]:
    data_transforms_train = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        # # A.VerticalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1, 
            p=0.5
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            sigma_limit=(0.1, 2.0),
            p=0.5
        ),
        A.Resize(height=512, width=512),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
else:
    data_transforms_train = A.Compose([
        A.Resize(height=512, width=512),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# Create transform pipeline that handles both image and mask
data_transforms_validation = A.Compose([
    A.Resize(height=512, width=512),
    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Initialize dataset with transforms
train_taco_dataset = TacoDatasetMask2Former(
    annotations_file="data/train_annotations.json",
    img_dir="data/images",
    processor=processor,
    transforms=data_transforms_train
)

# Initialize dataset with transforms
validation_taco_dataset = TacoDatasetMask2Former(
    annotations_file="data/validation_annotations.json",
    img_dir="data/images",
    processor=processor,
    transforms=data_transforms_validation
)

idx2class = train_taco_dataset.idx2class

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"].float() / 255.0 for example in batch])  # Convert to float and normalize
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    original_images = [example["original_image"] for example in batch]
    original_masks = [example["original_mask"] for example in batch]
    image_ids = [example["image_id"] for example in batch]
    inst2class = [example["inst2class"] for example in batch]

    #print(f"class_labels_collate_fn")
    #[print(example["class_labels"]) for example in batch]
    

    #for batch_index in range(len(batch)):
    #    for idx, mask in enumerate(mask_labels[batch_index]):
    #        print("Visualizing mask for:", idx2class[batch[batch_index]["class_labels"][idx].item()])
    #        visual_mask = (mask.bool().numpy() * 255).astype(np.uint8)
    #        img = Image.fromarray(visual_mask)
    #        img.show()

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
        "original_images": original_images,
        "original_masks": original_masks,
        "image_id": image_ids,
        "inst2class":inst2class
    }

train_loader = DataLoader(train_taco_dataset, 
                              batch_size = h_params["batch_size"],
                              num_workers = h_params["num_workers"],
                              shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=True)

validation_loader = DataLoader(validation_taco_dataset,
                               batch_size=h_params["batch_size"],
                               num_workers = h_params["num_workers"],
                               # shuffle=True,
                               collate_fn=collate_fn,
                               pin_memory=True)

# We specify ignore_mismatched_sizes=True to replace the already fine-tuned classification head by a new one
# model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade", # 101M params
#                                                           id2label=idx2class,
#                                                           ignore_mismatched_sizes=True)

# Define model configuration
# model_config = MaskFormerConfig.from_pretrained(
#     h_params["model_name"],
#     num_labels=len(idx2class),
#     output_hidden_states=True,
#     output_attentions=True,
#     use_auxiliary_loss=True,
#     id2label=idx2class
# )

print(f"len_idx2class: {len(idx2class)}")

# Load model with configuration
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    h_params["model_name"],
    num_labels=len(idx2class),
    #config=model_config,
    ignore_mismatched_sizes=True
)

model.config.backbone_config.id2label = idx2class
model.config.backbone_config.label2id = {v: k for k, v in idx2class.items()}

# Ensure output hidden states are enabled
model.config.output_hidden_states = True
model.config.use_auxiliary_loss = True

if h_params["load_checkpoint"]:
    checkpoint_path = h_params["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic", # 47 M params
#                                                             id2label=idx2class,
#                                                             ignore_mismatched_sizes=True)

# Freeze backbone's parameters from model.model.pixel_level_module.encoder.model.encoder.layers.0 to ...layers.2
#print("Model loaded...")
#print(model)

if(h_params["backbone_freeze"]):
    for param in model.model.pixel_level_module.encoder.parameters():
        param.requires_grad = False
else:
    for param in model.model.pixel_level_module.encoder.parameters():
        param.requires_grad = False
    # Unfreeze the last 3 layers of the encoder
    for param in model.model.pixel_level_module.encoder.encoder.layers[-1].parameters():
        param.requires_grad = True



# Freeze backbone's encoder parameters
# for param in model.model.pixel_level_module.parameters():
#     param.requires_grad = False

# Print the number of parameters for each model component
#print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
#print(f"Number of parameters in the pixel_level_module: {sum(p.numel() for p in model.model.pixel_level_module.parameters())}")
#print(f"Number of parameters in the pixel_level_module.encoder: {sum(p.numel() for p in model.model.pixel_level_module.encoder.parameters())}")
#print(f"Number of parameters in the pixel_level_module.encoder.model: {sum(p.numel() for p in model.model.pixel_level_module.encoder.model.parameters())}") # + hidden_states_norms
#print(f"Number of parameters in the pixel_level_module.encoder.hidden_states_norms: {sum(p.numel() for p in model.model.pixel_level_module.hidden_states_norms.parameters())}")
#print(f"Number of parameters in the pixel_level_module.decoder: {sum(p.numel() for p in model.model.pixel_level_module.decoder.parameters())}")
#print(f"Number of parameters in the transformer_module: {sum(p.numel() for p in model.model.transformer_module.parameters())}")
#print(f"Number of parameters in the class_predictor: {sum(p.numel() for p in model.class_predictor.parameters())}")
#print(f"Number of parameters in the mask_embedder: {sum(p.numel() for p in model.mask_embedder.parameters())}")
#print(f"Number of parameters in the matcher: {sum(p.numel() for p in model.matcher.parameters())}")
#print(f"Number of parameters in the criterion: {sum(p.numel() for p in model.criterion.parameters())}")

#for name, p in model.named_parameters():
#    print(name, p.requires_grad)

# batch = next(iter(train_dataloader))


model.to(device)

optimizer=torch.optim.AdamW(model.parameters(),
                            lr=h_params["learning_rate"],
                            weight_decay=h_params["weight_decay"])
# Load optimizer state
if h_params["load_checkpoint"]:
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    optimizer.load_state_dict(optimizer_state_dict)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)


def train_one_epoch():
    """
    Function to train one epoch of Mask2Former model.

    Returns:
        tuple: A tuple containing the average training loss and mean IoU.
    """
    model.train()
    losses_avg = 0
    total_samples = 0
    len_dataset = len(train_loader)
    
    for idx, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        try:
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
                pixel_mask=batch["pixel_mask"].to(device)
            )
        except RuntimeError as e:
            print(f"Error in batch {idx}: {e}")
            continue
        
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = batch["pixel_values"].size(0)
        losses_avg += loss.item() * batch_size
        total_samples += batch_size

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
    losses_avg = losses_avg / total_samples
    return losses_avg

metric_val = evaluate.load("mean_iou")

def validation_one_epoch():
    """
    Function to validate one epoch of Mask2Former model.

    Returns:
        tuple: A tuple containing the average validation loss and mean IoU.
    """
    model.eval()
    losses_avg=0
    total_samples = 0
    len_dataset=len(validation_loader)  
    # for  batch, data in enumerate(validation_loader):
    for idx, batch in enumerate(tqdm(validation_loader)):
        # print(f"Validation idx: {idx}")
        # images,targets=data            
        # images=list(image.to(device) for image in images)   
        # targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]             
        with torch.no_grad():
            try:
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(device) for labels in batch["class_labels"]],
                    pixel_mask=batch["pixel_mask"].to(device)
                )
            except RuntimeError as e:
                print(f"Error in batch {idx}: {e}")
                continue
            loss = outputs.loss
            # loss_dict = model(images, targets)
            # losses = sum(loss for loss in loss_dict.values())
    
            # loss_dict_printable = {k: f"{v.item():.2f}" for k, v in loss_dict.items()}      
            # print(f"[{batch}/{len_dataset}] total loss: {losses.item():.2f} losses: {loss_dict_printable}")     
            # losses_avg+=losses.item()    
            batch_size = batch["pixel_values"].size(0)
            losses_avg += loss.item() * batch_size
            total_samples += batch_size

        # Add memory cleanup after each batch
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        target_sizes = [(mask.shape[1], mask.shape[0]) for mask in batch['original_masks']]
        pred_maps = processor.post_process_instance_segmentation(
            outputs, target_sizes=target_sizes, threshold=0.5
        )
        #print("Predictions:")
        #print(outputs.masks_queries_logits)
        #print([np.unique(pred_map["segmentation"]) for pred_map in pred_maps])
        #print(batch['original_masks'])
        #print(pred_maps)
        # Resize predicted masks to match ground truth masks
        resized_pred_maps = []
        for pred_map, target_size in zip(pred_maps, target_sizes):
            pred_mask = pred_map["segmentation"].cpu().numpy()
            pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
            #pred_mask_resized = pred_mask_pil.resize(target_size, Image.NEAREST)
            pred_mask_resized = pred_mask_pil
            resized_pred_maps.append(np.array(pred_mask_resized))

        metric_val.add_batch(references=batch['original_masks'], predictions=resized_pred_maps)

    losses_avg = losses_avg / total_samples
    metrics_result = metric_val.compute(num_labels=len(idx2class), ignore_index=255, reduce_labels=True)
    # iou = 0
    return losses_avg, metrics_result
    


### START TRAINING
print("STARTING TRAINING")
train_loss=[]
validation_loss=[]
# Add variables to track best validation loss
best_val_loss = float('inf')
for epoch in range(1,h_params["num_epochs"]+1):
    losses_avg_train = train_one_epoch()
    losses_avg_validation, metrics_val = validation_one_epoch()
    print(f"TRAINING epoch[{epoch}/{h_params['num_epochs']}]: avg. loss: {losses_avg_train:.3f}")
    print(f"VALIDATION epoch[{epoch}/{h_params['num_epochs']}]: avg. loss:{ losses_avg_validation:.3f}")  
    train_loss.append(losses_avg_train)
    validation_loss.append(losses_avg_validation)

    save_model(model, epoch, optimizer, idx2class, os.path.join(results_dir, f"mask2former_{epoch}.pth"))    
    
    writer.add_scalar('Segmentation/train_loss', losses_avg_train, epoch)
    writer.add_scalar('Segmentation/val_loss', losses_avg_validation, epoch)
    writer.add_scalar('Segmentation/learning_rate', optimizer.param_groups[0]['lr'], epoch)

    for key, value in metrics_val.items():
        if isinstance(value, np.ndarray):
            writer.add_scalars(f'Segmentation/val_{key}', {str(i):v for i, v in enumerate(value)}, epoch)
        else:
            writer.add_scalar(f'Segmentation/val_{key}', value, epoch)
    
    # Save best model:
    if h_params["backup_best_model"]:
        if losses_avg_validation < best_val_loss:
            best_val_loss = losses_avg_validation
            save_model(model, epoch, optimizer, idx2class, os.path.join(results_dir, 'best_mask2former_model.pth'))
        # Update in training loop after validation
        scheduler.step(losses_avg_validation)

        # Call the script to upload the model checkpoint
        os.system(f"bash gcp_utils/upload_model_checkpoint.sh {os.path.join(results_dir, 'best_model.pth')} Mask2Former TACO")
        # Call the script to upload the logs
        os.system(f"bash gcp_utils/upload_model_checkpoint.sh {logdir} Mask2Former TACO")

print("Final train loss:\n")
print(train_loss)

print("Final validation loss:\n")
print(validation_loss)

### START EVALUATION
# print("STARING EVALUATION")
# test_taco_dataset=TacoDatasetMask2Former(annotations_file="data/test_annotations.json",
#                                     img_dir="data/images",
#                                     processor=processor,
#                                     transforms=data_transforms_validation)
# idx2class = test_taco_dataset.idx2class
# num_classes = len(idx2class)

# test_loader=DataLoader(test_taco_dataset,
#                        shuffle=False,
#                        batch_size=h_params["batch_size"], 
#                        num_workers=h_params["num_workers"],
#                        collate_fn=collate_fn)

# model.eval()
# metrics_dict = {}
# with torch.no_grad():
#     for idx, batch in enumerate(tqdm(test_loader)):
#         try:
#             outputs = model(
#                 pixel_values=batch["pixel_values"].to(device),
#                 mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
#                 class_labels=[labels.to(device) for labels in batch["class_labels"]]
#             )
#         except RuntimeError as e:
#             print(f"Error in batch {idx}: {e}")
#             continue

# print("Final test accuracy:\n")
# print(f"Metrics: {metrics}")
