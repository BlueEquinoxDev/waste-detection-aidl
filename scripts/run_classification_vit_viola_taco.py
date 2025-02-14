from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
from utilities.config_utils import TaskType
from custom_datasets.taco_dataset_vit import TacoDatasetViT
from custom_datasets.viola77_dataset import Viola77Dataset
from custom_datasets.taco_viola_dataset_vit import TacoViolaDatasetViT
# from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from torchvision.transforms import v2 as transforms
import os
from datasets import load_dataset, concatenate_datasets
from utilities.compute_metrics import create_compute_metrics
import numpy as np
import json

# Load the feature extractor and model
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Define data transforms
data_transforms_train = transforms.Compose([
    transforms.ToImage(),  # To tensor is deprecated
    transforms.ToDtype(torch.uint8, scale=True),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(0.5), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

data_transforms_test = transforms.Compose([
    transforms.ToImage(),  # To tensor is deprecated,
    transforms.ToDtype(torch.float32, scale=True),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the Viola dataset =================================================================
viola_dataset = load_dataset("viola77data/recycling-dataset", split="train")

# Split dataset into training, validation, and test sets
train_test_viola = viola_dataset.train_test_split(test_size=0.2)
val_test_viola = train_test_viola["test"].train_test_split(test_size=0.5)

train_dataset_viola = train_test_viola["train"]
val_dataset_viola = val_test_viola["train"]
test_dataset_viola = val_test_viola["test"]

# Create datasets with transforms
train_dataset_viola = Viola77Dataset(train_dataset_viola, transform=data_transforms_train)
val_dataset_viola = Viola77Dataset(val_dataset_viola, transform=data_transforms_test)
test_dataset_viola = Viola77Dataset(test_dataset_viola, transform=data_transforms_test)

# Obtain the categories form viola dataset
classes = viola_dataset.features['label'].names

# Load the TACO dataset =================================================================
train_annotations_file_taco = os.path.join("data", "train_annotations.json")
val_annotations_file_taco = os.path.join("data", "validation_annotations.json")
test_annotations_file_taco = os.path.join("data", "test_annotations.json")

# Prepare the mapping
with open("data/taco39viola11_categories.json", "r") as f:
    categories_taco_viola = json.load(f)
mapping = { cat["id"]: cat["super_id"] for cat in categories_taco_viola }

# Create datasets with transforms
train_dataset_taco = TacoViolaDatasetViT(annotations_file=train_annotations_file_taco, img_dir="data/images", transform=data_transforms_train, classes=classes, mapping=mapping)
val_dataset_taco = TacoViolaDatasetViT(annotations_file=val_annotations_file_taco, img_dir="data/images", transform=data_transforms_test, classes=classes, mapping=mapping)
test_dataset_taco = TacoViolaDatasetViT(annotations_file=test_annotations_file_taco, img_dir="data/images", transform=data_transforms_test, classes=classes, mapping=mapping)

# Concatenate the datasets =================================================================
train_dataset = torch.utils.data.ConcatDataset([train_dataset_viola, train_dataset_taco])
val_dataset = torch.utils.data.ConcatDataset([val_dataset_viola, val_dataset_taco])
test_dataset = torch.utils.data.ConcatDataset([test_dataset_viola, test_dataset_taco])

print("Printing the train dataset")
print(train_dataset)

# Get number of classes and label names from dataset
num_classes = len(train_dataset_taco.idx_to_cluster_class)
label_names = list(train_dataset_taco.idx_to_cluster_class.values())
id2label = train_dataset_taco.idx_to_cluster_class
label2id = train_dataset_taco.cluster_class_to_idx
print(f"Number of classes: {num_classes} | Label names: {label_names}")

# Initialize model with number of labels
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Create compute_metrics function with label names
metrics_function = create_compute_metrics(label_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=metrics_function,
    processing_class=feature_extractor,  # Changed from tokenizer to processing_class
)

# Train the model
trainer.train()

# Evaluate the model
val_results = trainer.evaluate()
print(f'Evaluation results: {val_results}')
