from torch.utils.data import DataLoader
from transformers import MaskFormerImageProcessor, MaskFormerConfig
import numpy as np
from custom_datasets.taco_dataset_mask2former import TacoDatasetMask2Former 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import MaskFormerForInstanceSegmentation, AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utilities.save_model import save_model
from utilities.maskformer_display_sample_results import display_sample_results
import os


h_params ={
    "batch_size": 10,
    "num_workers": 2,
    "num_epochs": 20,
    "learning_rate": 5e-5,
    "weight_decay": 1e-2,
    "model_name": "facebook/mask2former-swin-tiny-ade-semantic",
    "backbone_freeze": False,
    "augmentation": False,
    "backup_best_model": True
}

# experiment_name contains model name, backbone_freeze, augmentation
experiment_name = "seg-taco-"+h_params["model_name"].split("/")[1]+"-"+str(h_params["backbone_freeze"])+"_backbone_freeze-"+str(h_params["augmentation"])+"_augmentation"
print("Experiment name: ", experiment_name)

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

# Initialize processor with proper configuration
processor = AutoImageProcessor.from_pretrained(
    h_params["model_name"],
    do_resize=True,
    do_rescale=False,  # Disable rescaling since we handle it in transforms
    do_normalize=False  # Disable normalization since we handle it in transforms
)

# Create transform pipeline that handles both image and mask
data_transforms_train = A.Compose([
    # A.RandomRotate90(p=0.5),
    # A.HorizontalFlip(p=0.5),
    # # A.VerticalFlip(p=0.5),
    # A.ColorJitter(
    #     brightness=0.2, 
    #     contrast=0.2, 
    #     saturation=0.2, 
    #     hue=0.1, 
    #     p=0.5
    # ),
    # A.GaussianBlur(
    #     blur_limit=(3, 7),
    #     sigma_limit=(0.1, 2.0),
    #     p=0.5
    # ),
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Create transform pipeline that handles both image and mask
data_transforms_validation = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# Computing samples weights for WeightedRandomSampler
number_of_samples_by_category = {
    cat: len(train_taco_dataset.coco_data.catToImgs[cat]) 
    for cat in train_taco_dataset.idx2class.keys() 
    if cat != 0  # Exclude background class
}
print(f"Number of samples by category: {number_of_samples_by_category}")

number_of_samples = sum(number_of_samples_by_category.values())
print(f"Number of samples: {number_of_samples}")

# Handle categories with zero samples by setting their weight to a high value
category_weights = {
    cat: number_of_samples/max(value, 1)  # Use max(value, 1) to avoid division by zero
    for cat, value in number_of_samples_by_category.items()
}
print(f"Category weights: {category_weights}")

# Computing samples weights for WeightedRandomSampler
samples_weights = []

for idx in range(train_taco_dataset.len_dataset):
    weights_cat_in_image = []
    img_id = train_taco_dataset.index_to_imageId[idx]
    annotations = train_taco_dataset.coco_data.imgToAnns[img_id]
    
    for ann in annotations:
        # Map the original category_id to our remapped category using category_map
        mapped_category_id = train_taco_dataset.category_map[ann['category_id']]
        if mapped_category_id in category_weights:
            weights_cat_in_image.append(category_weights[mapped_category_id])
    
    # If no valid categories found, use default weight of 1.0
    if weights_cat_in_image:
        samples_weights.append(max(weights_cat_in_image))
    else:
        samples_weights.append(1.0)

sampler = torch.utils.data.WeightedRandomSampler(
    weights=samples_weights, 
    num_samples=train_taco_dataset.len_dataset, 
    replacement=True
)

train_loader = DataLoader(train_taco_dataset, 
                              batch_size = h_params["batch_size"],
                              num_workers = h_params["num_workers"],
                              shuffle=False, # Use WeightedRandomSampler instead
                              collate_fn=collate_fn,
                              pin_memory=True,
                              sampler=sampler)

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
model_config = MaskFormerConfig.from_pretrained(
    h_params["model_name"],
    num_labels=len(idx2class),
    output_hidden_states=True,
    output_attentions=True,
    use_auxiliary_loss=True,
    id2label=idx2class
)

# Load model with configuration
model = MaskFormerForInstanceSegmentation.from_pretrained(
    h_params["model_name"],
    config=model_config,
    ignore_mismatched_sizes=True
)

# Ensure output hidden states are enabled
model.config.output_hidden_states = True
model.config.use_auxiliary_loss = True

# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic", # 47 M params
#                                                             id2label=idx2class,
#                                                             ignore_mismatched_sizes=True)

# Freeze backbone's parameters from model.model.pixel_level_module.encoder.model.encoder.layers.0 to ...layers.2
if(h_params["backbone_freeze"]):
    for param in model.model.pixel_level_module.encoder.model.encoder.layers[:3].parameters():
        param.requires_grad = False

# Freeze backbone's encoder parameters
# for param in model.model.pixel_level_module.parameters():
#     param.requires_grad = False

# Print the number of parameters for each model component
print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
print(f"Number of parameters in the pixel_level_module: {sum(p.numel() for p in model.model.pixel_level_module.parameters())}")
print(f"Number of parameters in the pixel_level_module.encoder: {sum(p.numel() for p in model.model.pixel_level_module.encoder.parameters())}")
print(f"Number of parameters in the pixel_level_module.encoder.model: {sum(p.numel() for p in model.model.pixel_level_module.encoder.model.parameters())}") # + hidden_states_norms
#print(f"Number of parameters in the pixel_level_module.encoder.hidden_states_norms: {sum(p.numel() for p in model.model.pixel_level_module.hidden_states_norms.parameters())}")
print(f"Number of parameters in the pixel_level_module.decoder: {sum(p.numel() for p in model.model.pixel_level_module.decoder.parameters())}")
print(f"Number of parameters in the transformer_module: {sum(p.numel() for p in model.model.transformer_module.parameters())}")
print(f"Number of parameters in the class_predictor: {sum(p.numel() for p in model.class_predictor.parameters())}")
print(f"Number of parameters in the mask_embedder: {sum(p.numel() for p in model.mask_embedder.parameters())}")
print(f"Number of parameters in the matcher: {sum(p.numel() for p in model.matcher.parameters())}")
print(f"Number of parameters in the criterion: {sum(p.numel() for p in model.criterion.parameters())}")

# batch = next(iter(train_dataloader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer=torch.optim.AdamW(model.parameters(),
                            lr=h_params["learning_rate"],
                            weight_decay=h_params["weight_decay"])

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

def train_one_epoch():
    """
    Function to train 1 epoch of Mask R-CNN with periodic loss printing
    returns:
        -  avg loss in training
    """
    model.train()
    losses_avg = 0
    running_loss = 0  # For periodic printing
    total_samples = 0
    running_samples = 0  # For periodic printing
    len_dataset = len(train_loader)
    
    for idx, batch in enumerate(tqdm(train_loader)):
        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        try:
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )
        except RuntimeError as e:
            print(f"Error in batch {idx}: {e}")
            continue

        # Backward propagation
        loss = outputs.loss
        loss.backward()

        # Gradient clipping before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        batch_size = batch["pixel_values"].size(0)
        losses_avg += loss.item() * batch_size
        total_samples += batch_size
        
        # Track running loss for periodic printing
        running_loss += loss.item() * batch_size
        running_samples += batch_size

        # Print loss every 10 steps
        if (idx + 1) % 100 == 0:
            avg_running_loss = running_loss / running_samples
            print(f"\nStep [{idx + 1}/{len_dataset}] - Running Loss: {avg_running_loss:.4f}")
            # Reset running metrics
            running_loss = 0
            running_samples = 0

        # Optimization
        optimizer.step()

        # if idx > 100:
        #     display_sample_results(batch, outputs, processor, mask_threshold = 0.01)

        # # Add memory cleanup after each batch
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
    
    losses_avg = losses_avg / total_samples
    return losses_avg

def validation_one_epoch():
    """
    Function to validate 1 epoch of Mask R-CNN
    returns:
        -  avg loss in validation
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

    losses_avg = losses_avg / total_samples
    return losses_avg
    


### START TRAINING
print("STARTING TRAINING")
train_loss=[]
validation_loss=[]
# Add variables to track best validation loss
best_val_loss = float('inf')
for epoch in range(1,h_params["num_epochs"]+1):
    losses_avg_train=train_one_epoch()
    losses_avg_validation=validation_one_epoch()
    print(f"TRAINING epoch[{epoch}/{h_params['num_epochs']}]: avg. loss: {losses_avg_train:.3f}")
    print(f"VALIDATION epoch[{epoch}/{h_params['num_epochs']}]: avg. loss:{ losses_avg_validation:.3f}")  
    train_loss.append(losses_avg_train)
    validation_loss.append(losses_avg_validation)
    # save_model(model, epoch, optimizer, idx2class, results_dir)    
    writer.add_scalar('Segmentation/train_loss', losses_avg_train, epoch)
    writer.add_scalar('Segmentation/val_loss', losses_avg_validation, epoch)
    writer.add_scalar('Segmentation/learning_rate', optimizer.param_groups[0]['lr'], epoch)
    # Save best model:
    if h_params["backup_best_model"]:
        if losses_avg_validation < best_val_loss:
            best_val_loss = losses_avg_validation
            save_model(model, epoch, optimizer, idx2class, os.path.join(results_dir, 'best_model.pth'))
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