from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
from utilities.config_utils import TaskType
from custom_datasets.taco_dataset_vit import TacoDatasetViT
from custom_datasets.viola77_dataset import Viola77Dataset
# from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from torchvision.transforms import v2 as transforms
import os
from datasets import load_dataset
from utilities.compute_metrics import create_compute_metrics
import numpy as np

# Choose dataset
DATASET = "VIOLA77" # "TACO" or "VIOLA77"

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


if DATASET == "TACO":

    train_annotations_file = os.path.join("data", "train_annotations.json")
    val_annotations_file = os.path.join("data", "validation_annotations.json")
    test_annotations_file = os.path.join("data", "test_annotations.json")

    subset_classes = {
        "Bottle": [4, 5, 6],
    # "Carton": [14, 18],  # Cluster IDs 14 and 18
        "Cup": [20, 21, 22, 23, 24],
        "Can": [10, 12],
        "Plastic film": [36]
    }

    # Load the TACO dataset
    train_dataset = TacoDatasetViT(annotations_file=train_annotations_file, img_dir="data", transforms=data_transforms_train, subset_classes = subset_classes)
    val_dataset = TacoDatasetViT(annotations_file=val_annotations_file, img_dir="data", transforms=data_transforms_test, subset_classes = subset_classes)
    test_dataset = TacoDatasetViT(annotations_file=test_annotations_file, img_dir="data", transforms=data_transforms_test, subset_classes = subset_classes)

elif DATASET == "VIOLA77":
    # Load the dataset
    dataset = load_dataset("viola77data/recycling-dataset", split="train")
    print(dataset)

    # Split dataset into training, validation, and test sets
    train_test = dataset.train_test_split(test_size=0.2)
    val_test = train_test["test"].train_test_split(test_size=0.5)

    train_dataset = train_test["train"]
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    # Create datasets with transforms
    train_dataset = Viola77Dataset(train_dataset, transform=data_transforms_train)
    val_dataset = Viola77Dataset(val_dataset, transform=data_transforms_test)
    test_dataset = Viola77Dataset(test_dataset, transform=data_transforms_test)

# Get number of classes and label names from dataset
num_classes = len(train_dataset.idx_to_cluster_class)
label_names = list(train_dataset.idx_to_cluster_class.values())
print(f"Number of classes: {num_classes} | Label names: {label_names}")

# Initialize model with number of labels
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=num_classes,
    id2label=train_dataset.idx_to_cluster_class,
    label2id=train_dataset.cluster_class_to_idx,
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