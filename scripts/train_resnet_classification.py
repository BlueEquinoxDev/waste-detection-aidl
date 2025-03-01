import os
from datetime import datetime

from datasets import load_dataset
from utilities.compute_metrics import create_compute_metrics
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
from tqdm import tqdm
from utilities.config_utils import TaskType
from custom_datasets.viola_dataset_resnet import Viola77DatasetResNet
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from torchvision.transforms import v2 as transforms
from torchvision import models, transforms

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

EXPERIMENT_NAME = "cls-resnet-viola"

# Define data transforms
train_transform = transforms.Compose([
    transforms.RandomChoice([  # Randomly apply ONE transformation
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop & resize
        transforms.RandomHorizontalFlip(p=0.5),  # Always flip horizontally (when chosen)
        transforms.RandomRotation(degrees=15),  # Rotate randomly
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Color changes
        transforms.GaussianBlur(kernel_size=3),  # Blurring effect
    ]),
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
train_dataset = Viola77DatasetResNet(train_dataset, transform=train_transform)
val_dataset = Viola77DatasetResNet(val_dataset, transform=val_test_transform)
test_dataset = Viola77DatasetResNet(test_dataset, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

sample = train_dataset[0]
print(type(sample['pixel_values']))  # Should be <class 'torch.Tensor'>

# Get number of classes and label names from dataset
num_classes = len(train_dataset.idx_to_cluster_class)
label_names = list(train_dataset.idx_to_cluster_class.values())
print(f"Number of classes: {num_classes} | Label names: {label_names}")



# Initialize model with number of labels
model = models.resnet50(pretrained=True)

model.fc = nn.Sequential(
    nn.Dropout(0.26740334504821867),
    nn.Linear(model.fc.in_features, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# Optim and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=2.0667521318354856e-05)
criterion = torch.nn.CrossEntropyLoss()

# Create compute_metrics function with label names
metrics_function = create_compute_metrics(label_names)

logdir = os.path.join("logs", f"{EXPERIMENT_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

# Initialize Tensorboard Writer with the previous folder 'logdir'
writer=SummaryWriter(log_dir=logdir)


# Metrics
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

# # Checkpoint
# print("Train labels unique values:", train_df['label'].unique())
# print("Val labels unique values:", val_df['class_name'].unique())
# print("Test labels unique values:", test_df["class_name"].unique())

# Training (afegim checkpoints)
num_epochs = 15
best_val_acc = 0.0

sample = train_dataset[0]
print(type(sample['pixel_values']))  # Should be <class 'torch.Tensor'>

if __name__ == "__main__":
    # Everything related to training should go inside this block
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = batch['pixel_values']
            labels = batch['labels']
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy * 100:.2f}%")

    # Validation
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        all_labels = []
        all_predictions = []
        all_raw_outputs = []  # Store raw outputs before argmax

        with torch.no_grad():
            for batch in val_loader:
                images = batch['pixel_values']
                labels = batch['labels']
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_raw_outputs.extend(outputs.cpu().numpy())  # Save raw outputs

        val_accuracy = correct_preds / total_preds
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), "./model/best_resnet50.pth")
                print(f"New best model saved with Accuracy: {val_accuracy * 100:.2f}%")

        print("Training complete! Best model saved as 'best_resnet50.pth'.")

    results_dir = "./metrics/resnet"
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    # KPIs
    all_probabilities = F.softmax(torch.tensor(all_raw_outputs), dim=1).numpy()
    conf_matrix = confusion_matrix(all_labels, np.argmax(all_probabilities, axis=1))
    f1 = f1_score(all_labels, np.argmax(all_probabilities, axis=1), average="weighted")
    auc = roc_auc_score(all_labels, all_probabilities, multi_class="ovr")

    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nF1 Score:", f1)
    print("\nAUC Score:", auc)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    Conf_mtx = os.path.join(results_dir, "TRAIN_confusion_matrix.png")
    plt.savefig(Conf_mtx)
    plt.show()

    # Accuracy & Loss
    #epochs_range = range(1, num_epochs + 1)  # Original Line
    epochs_range = range(1, len(train_losses) + 1)  # Corrected Line to match the length of train_losses

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Training Accuracy", marker="o")
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    Loss_Acc_Plot = os.path.join(results_dir, "TRAIN_metrics.png")
    plt.savefig(Loss_Acc_Plot)
    plt.show()

    print("Training complete! Model and class mapping saved. Confusion Matrix and Metrics charts generated.")

