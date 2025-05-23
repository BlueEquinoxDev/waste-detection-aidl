import os
import json
from datetime import datetime
import argparse
import random

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
from torchvision import models, transforms

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--enhanced_hparams', action='store_true', help='Use optimized hyperparameters from Optuna')
parser.add_argument('--lr', type=float, default=2.0667521318354856e-05, help='Learning rate for training')
parser.add_argument('--dropout', type=float, default=0.26740334504821867, help='Dropout rate for the model')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training')
args = parser.parse_args()

EXPERIMENT_NAME = "cls-resnet-viola"

# Set the seed 
SEED = 42

# Define data transforms
train_transform = transforms.Compose([
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

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = load_dataset("viola77data/recycling-dataset", split="train")
train_test = dataset.train_test_split(test_size=0.2, seed= SEED)
val_test = train_test["test"].train_test_split(test_size=0.5, seed = SEED)

train_dataset = Viola77DatasetResNet(train_test["train"], transform=train_transform)
val_dataset = Viola77DatasetResNet(val_test["train"], transform=val_test_transform)
test_dataset = Viola77DatasetResNet(val_test["test"], transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.idx_to_cluster_class)
label_names = list(train_dataset.idx_to_cluster_class.values())

# Execute Optuna if enhanced_hparams is set to True
hparams_path = "./utilities/hparams.json"
if args.enhanced_hparams:
    os.system("python -m utilities.optuna_resnet_hparams")
    
if os.path.exists(hparams_path) and args.enhanced_hparams == True:
    with open(hparams_path, "r") as f:
        hparams = json.load(f)
else:
    hparams = {
        "lr": args.lr, 
        "dropout": args.dropout,
        "epochs": args.epochs
    }

# Initialize model
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(hparams["dropout"]),
    nn.Linear(model.fc.in_features, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if args.enhanced_hparams:
    if "optimizer" in hparams:
        if hparams["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
        elif hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=hparams["lr"], momentum=0.9)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=hparams["lr"])
    else:
        raise KeyError("Missing 'optimizer' key in hparams.json. Ensure Optuna is saving it correctly.")
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

criterion = torch.nn.CrossEntropyLoss()
logdir = os.path.join("logs", f"{EXPERIMENT_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
metrics_function = create_compute_metrics(label_names, logdir)
writer = SummaryWriter(log_dir=logdir)

num_epochs = hparams.get("epochs",15)
best_val_acc = 0.0
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

print(hparams["dropout"], hparams["lr"])

if __name__ == "__main__":
    print("Starting training...")
    for epoch in range(hparams.get("epochs", 15)):
        model.train()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        for batch in tqdm(train_loader):
            images, labels = batch['pixel_values'].to(device), batch['labels'].to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct_preds += (outputs.argmax(1) == labels).sum().item()
            total_preds += labels.size(0)
        train_accuracy = correct_preds / total_preds
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy * 100:.2f}%")


        model.eval()
        val_loss, correct_preds, total_preds = 0.0, 0, 0
        all_labels = []
        all_predictions = []
        all_raw_outputs = []
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch['pixel_values'].to(device), batch['labels'].to(device).long()
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                correct_preds += (outputs.argmax(1) == labels).sum().item()
                total_preds += labels.size(0)
                
                all_labels.extend(labels.cpu().numpy())
                all_raw_outputs.extend(outputs.cpu().numpy())  # Save raw outputs

        
        val_accuracy = correct_preds / total_preds
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracies[-1] > best_val_acc:
            best_val_acc = val_accuracies[-1]
            torch.save(model.state_dict(), "./model/best_resnet50.pth")
            print(f"New best model saved with Accuracy: {best_val_acc * 100:.2f}%")

    print("Training complete! Best model saved.")
    results_dir = "./metrics/resnet/train"
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

