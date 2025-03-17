import os
import csv
import random
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO

from datasets import load_dataset
from transformers import ViTForImageClassification
from torchvision import transforms
from custom_datasets.viola77_dataset import Viola77Dataset

from model.waste_vit import WasteViT

# -------------------- Configuration --------------------
DATASET_NAME = "viola77data/recycling-dataset"
BASE_DIR = "data/viola_dataset"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations.csv")
UPDATED_ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations_updated.csv")
EXPERIMENT_NAME = "cls-vit-viola"
NUM_EPOCHS = 15
BATCH_SIZE = 32
SEED = 42

# Class mapping for annotation updates
# CLASS_MAPPING = {
#     0: "aluminium",
#     1: "batteries",
#     2: "cardboard",
#     3: "disposable_plates",
#     4: "glass",
#     5: "hard_plastic",
#     6: "paper",
#     7: "paper_towel",
#     8: "polystyrene",
#     9: "soft_plastics",
#     10: "takeaway_cups",
# }

def create_transforms():
    data_transforms_train = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.GaussianBlur(kernel_size=3),
        ]),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transforms_train, data_transforms_test

def setup_experiment():
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logs_dir = os.path.join("logs", f"{EXPERIMENT_NAME}-{timestamp}")
    results_dir = os.path.join("results", f"{EXPERIMENT_NAME}-{timestamp}")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logs_dir)
    print(f"Logs directory: {logs_dir}")
    print(f"Results directory: {results_dir}")
    return logs_dir, results_dir, writer

def create_model(num_classes):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=num_classes
    )
    model.classifier = nn.Sequential(
        nn.Dropout(0.48699113095794067),
        nn.Linear(model.config.hidden_size, num_classes)
    )
    return model

def train_model(model, train_loader, val_loader, experiment_name, results_dir, writer, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0023280870643464266)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_model_name = None
    all_labels, all_raw_outputs = None, None

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        train_accuracy = correct_preds / total_preds
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy*100:.2f}%")

        # Log training metrics
        writer.add_scalar('Loss/train', train_losses[-1], epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        all_labels = []
        all_raw_outputs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_raw_outputs.extend(outputs.cpu().numpy())

        val_accuracy = correct_preds / total_preds
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_name = f"{experiment_name}-{datetime.now().strftime('%Y%m%d')}.pth"
            model_path = os.path.join(results_dir, best_model_name)
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with Accuracy: {val_accuracy*100:.2f}%")

        # Log validation metrics
        writer.add_scalar('Loss/val', val_losses[-1], epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Optionally log learning rate
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    print(f"Training complete! Best model saved as: {best_model_name}")
    return best_model_name, all_labels, all_raw_outputs

def evaluate_model(all_labels, all_raw_outputs, class_names, results_dir):
    probabilities = F.softmax(torch.tensor(all_raw_outputs), dim=1).numpy()
    pred_labels = np.argmax(probabilities, axis=1)
    conf_matrix = confusion_matrix(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels, average="weighted")
    auc = roc_auc_score(all_labels, probabilities, multi_class="ovr")

    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nF1 Score:", f1)
    print("\nAUC Score:", auc)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    conf_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    print(f"Confusion matrix saved to: {conf_matrix_path}")

def main():
    
    # Load the dataset
    dataset = load_dataset("viola77data/recycling-dataset", split="train")
    print(dataset)

    # Split dataset into training, validation, and test sets
    train_test = dataset.train_test_split(test_size=0.2, seed=SEED)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=SEED)

    train_dataset = train_test["train"]
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    # Set up transforms
    data_transforms_train, data_transforms_test = create_transforms()

    # Create datasets with transforms
    train_dataset = Viola77Dataset(train_dataset, transform=data_transforms_train)
    val_dataset = Viola77Dataset(val_dataset, transform=data_transforms_test)
    test_dataset = Viola77Dataset(test_dataset, transform=data_transforms_test)
    
    # Setup experiment directories and TensorBoard writer
    logs_dir, results_dir, writer = setup_experiment()
    
    # Ensure consistent label mapping: reassign labels to be contiguous integers
    annotations = pd.read_csv(UPDATED_ANNOTATIONS_FILE)
    unique_labels = sorted(annotations["label"].unique())
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    # print("Class Mapping:", label_mapping)
    
    # for df in [train_df, val_df, test_df]:
    #     df["label"] = df["label"].map(label_mapping)
    num_classes = len(label_mapping)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create and prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # checkpoint_path = os.path.join(results_dir, "best_vit_viola.pth")
    checkpoint_path = None

    num_classes = len(train_dataset.idx_to_cluster_class)
    label_names = list(train_dataset.idx_to_cluster_class.values())
    id2label = {idx: label for idx, label in train_dataset.idx_to_cluster_class.items()}
    label2id = train_dataset.cluster_class_to_idx

    # Create an instance of the model with the checkpoint
    model = WasteViT(num_classes=num_classes, id2label=id2label, label2id=label2id, checkpoint=checkpoint_path)
    model.to(device)
    
    # Train the model
    best_model_name, all_labels, all_raw_outputs = train_model(model, train_loader, val_loader, EXPERIMENT_NAME, results_dir, writer, device)
    
    # Evaluate the model
    # Use the class names in the order of label mapping for the confusion matrix
    # class_names = [CLASS_MAPPING[label] for label in unique_labels]
    # define class_names using train_dataset.cluster_class_to_idx
    class_names = [train_dataset.idx_to_cluster_class[idx] for idx in range(len(train_dataset.idx_to_cluster_class))]

    evaluate_model(all_labels, all_raw_outputs, class_names, results_dir)

if __name__ == "__main__":
    main()
