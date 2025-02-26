import argparse
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
from datasets import load_dataset
from utilities.compute_metrics import create_compute_metrics
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
from model.waste_vit import WasteViT

# Parse arguments
parser = argparse.ArgumentParser(description='Select the dataset for model training')
parser.add_argument('--dataset', required=False, help='Dataset name', type=str, default="TACO5")

# Check if the given dataset is valid
valid_datasets = ["TACO5", "TACO28", "VIOLA", "TACO39VIOLA11"]
dataset_name = parser.parse_args().dataset
if dataset_name not in valid_datasets:
    raise ValueError(f"Dataset must be one of {valid_datasets}")

experiment_name = f"cls-vit-{dataset_name.lower()}"

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

if "TACO" in dataset_name and "VIOLA" not in dataset_name:

    train_annotations_file = os.path.join("data", "train_annotations.json")
    val_annotations_file = os.path.join("data", "validation_annotations.json")
    test_annotations_file = os.path.join("data", "test_annotations.json")

    if dataset_name == "TACO5":
        # read subset_classes from taco5_categories.json
        subset_classes_file = os.path.join("data", "taco5_categories.json")
        subset_classes = {}
        with open(subset_classes_file, "r") as f:
            subset_classes = json.load(f)
    elif dataset_name == "TACO28":
        # read subset_classes from taco28_categories.json
        subset_classes_file = os.path.join("data", "taco28_categories.json")
        subset_classes = {}
        with open(subset_classes_file, "r") as f:
            subset_classes = json.load(f) 

    # Load the TACO dataset
    train_dataset = TacoDatasetViT(annotations_file=train_annotations_file, img_dir="data/images", transforms=data_transforms_train, subset_classes = subset_classes)
    val_dataset = TacoDatasetViT(annotations_file=val_annotations_file, img_dir="data/images", transforms=data_transforms_test, subset_classes = subset_classes)
    test_dataset = TacoDatasetViT(annotations_file=test_annotations_file, img_dir="data/images", transforms=data_transforms_test, subset_classes = subset_classes)

elif dataset_name == "VIOLA":
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

elif dataset_name == "TACO39VIOLA11":
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


# After creating the concatenated dataset
if dataset_name == "TACO39VIOLA11":
    # Get number of classes and label mappings from one of the constituent datasets
    # Assuming both datasets use the same class mappings
    if isinstance(train_dataset.datasets[0], (TacoDatasetViT, Viola77Dataset)):
        base_dataset = train_dataset.datasets[0]
        num_classes = len(base_dataset.idx_to_cluster_class)
        label_names = list(base_dataset.idx_to_cluster_class.values())
        id2label = {idx: label for idx, label in base_dataset.idx_to_cluster_class.items()}
        label2id = base_dataset.cluster_class_to_idx
    else:
        raise ValueError("Unexpected dataset type")
else:
    # For single datasets
    num_classes = len(train_dataset.idx_to_cluster_class)
    label_names = list(train_dataset.idx_to_cluster_class.values())
    id2label = {idx: label for idx, label in train_dataset.idx_to_cluster_class.items()}
    label2id = train_dataset.cluster_class_to_idx

print(f"Number of classes: {num_classes} | Label names: {label_names}")

# model = WasteViT(num_classes=num_classes, id2label = id2label, label2id = label2id)
model = WasteViT(checkpoint="results/cls-vit-taco39viola11-20250218-200130/checkpoint-5620")

logdir = os.path.join("logs", f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
results_dir = os.path.join("results", f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

# Create compute_metrics function with label names
metrics_function = create_compute_metrics(label_names, logdir)

# Define training arguments
training_args = TrainingArguments(
    output_dir=results_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=logdir,
    logging_strategy="epoch",
    logging_steps=1,  # Log every 1 epoch  
    report_to=["tensorboard"],  # Enable tensorboard reporting
    load_best_model_at_end=True,  # Load the best model after training
    metric_for_best_model="accuracy",  # Define metric to track
    save_total_limit=3,  # Limit total saved checkpoints
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=metrics_function,
    processing_class=model.feature_extractor,  # Changed from tokenizer to processing_class
)

# Train the model
trainer.train()

# Evaluate the model
val_results = trainer.evaluate()
print(f'Evaluation results: {val_results}')