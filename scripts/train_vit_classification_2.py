import os
import csv
import datasets
from datasets import load_dataset, DownloadConfig
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import random

import os
import torch
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms

from PIL import Image
from tqdm import tqdm

from datetime import datetime

from transformers import ViTForImageClassification

import torch

print(torch.__version__)


# Define paths
dataset_name = "viola77data/recycling-dataset"
base_dir = "data/viola_dataset"
images_dir = os.path.join(base_dir, "images")
annotations_file = os.path.join(base_dir, "annotations.csv")

# Load dataset
dataset = load_dataset(dataset_name, split="train")  # Adjust split if needed

# Create directories
os.makedirs(images_dir, exist_ok=True)

# Open CSV file to store annotations
with open(annotations_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "label"])

    # Iterate through dataset
    for idx, data in enumerate(dataset):
        if "image" not in data or "label" not in data:
            print(f"Skipping entry {idx} due to missing keys.")
            continue

        image_data = data["image"]
        label = data["label"]

        # If images are URLs, download them
        if isinstance(image_data, str):
            response = requests.get(image_data)
            image = Image.open(BytesIO(response.content))
        else:
            image = image_data  # Already a PIL Image

        # Save image
        image_path = os.path.join(images_dir, f"image_{idx}.jpg")
        image.save(image_path)

        # Write annotation
        writer.writerow([image_path, label])

print(f"Images saved in: {images_dir}")
print(f"Annotations saved in: {annotations_file}")


####


# Class mapping
class_mapping = {
    0: "aluminium",
    1: "batteries",
    2: "cardboard",
    3: "disposable_plates",
    4: "glass",
    5: "hard_plastic",
    6: "paper",
    7: "paper_towel",
    8: "polystyrene",
    9: "soft_plastics",
    10: "takeaway_cups",
}

# Load annotations
df = pd.read_csv(annotations_file)

# Check if required columns exist
if "image_path" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV file must contain 'image_path' and 'label' columns.")

# Ensure labels are integers before mapping
df["label"] = df["label"].astype(int)

# Create a new column with mapped class names
df["class_name"] = df["label"].map(class_mapping)

# Save the updated CSV file
updated_annotations_file = os.path.join(base_dir, "annotations_updated.csv")
df.to_csv(updated_annotations_file, index=False)

# Count occurrences of each class
class_counts = Counter(df["class_name"])

# Convert to DataFrame for sorting and analysis
df_counts = pd.DataFrame(class_counts.items(), columns=["Class", "Count"])
df_counts = df_counts.sort_values(by="Count", ascending=False)

# Print dataset statistics
total_images = len(df)
num_classes = len(class_counts)

print("Dataset Summary:")
print(f"Total Images: {total_images}")
print(f"Number of Classes: {num_classes}")
print(df_counts)

# Plot class distribution
plt.figure(figsize=(10, 6))
plt.bar(df_counts["Class"], df_counts["Count"], color="skyblue")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution in Dataset")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save plot
plot_path = os.path.join(base_dir, "class_distribution.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.show()

print(f"Updated annotations saved to: {updated_annotations_file}")
print(f"Class distribution plot saved to: {plot_path}")


########################################

# Load annotations
df = pd.read_csv(updated_annotations_file)

# Ensure the dataset has the required columns
if "image_path" not in df.columns or "class_name" not in df.columns:
    raise ValueError("CSV file must contain 'image_path' and 'class_name' columns.")

# Split dataset (80% train, 10% val, 10% test)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["class_name"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["class_name"], random_state=42)

# Save the splits
train_df.to_csv(os.path.join(base_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(base_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(base_dir, "test.csv"), index=False)

# Print split summary
print(f"Dataset split completed!")
print(f"Train set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")
print(f"Test set: {len(test_df)} images")

print(f"Train split saved to: {os.path.join(base_dir, 'train.csv')}")
print(f"Validation split saved to: {os.path.join(base_dir, 'val.csv')}")
print(f"Test split saved to: {os.path.join(base_dir, 'test.csv')}")


########################################

num_images = 25
random_indices = random.sample(range(len(train_df)), num_images)
original_images = [train_df.iloc[i][['image_path', 'class_name']].tolist() for i in random_indices]
print(original_images[:])

original_images_pil = [Image.open(img[0]).convert("RGB") for img in original_images]

# Create a 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        idx = i * 5 + j  # Image index
        axes[i, j].imshow(original_images_pil[idx])
        axes[i, j].set_title(original_images[idx][1])
        axes[i, j].axis("off")

plt.tight_layout()
# Save plot
plot_path_train = os.path.join(base_dir, "example_train_images.png")
plt.savefig(plot_path_train, bbox_inches="tight")


########################################

experiment_name = f"cls-vit-viola-Marti"
logs_dir = os.path.join("logs", f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
os.makedirs(logs_dir, exist_ok=True)

results_dir = os.path.join("results", f"{experiment_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
os.makedirs(results_dir, exist_ok=True)

writer = SummaryWriter(log_dir=logs_dir)

train_file = os.path.join(base_dir, "train.csv")
val_file = os.path.join(base_dir, "val.csv")
test_file = os.path.join(base_dir, "test.csv")

annotations_df = pd.read_csv(updated_annotations_file)
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

# Class names
class_names = annotations_df["label"].unique()
class_mapping = {int(class_name): idx for idx, class_name in enumerate(class_names)}

# class labels -> numerical labels
train_df["label"] = train_df["label"].map(class_mapping)
val_df["label"] = val_df["label"].map(class_mapping)
test_df["label"] = test_df["label"].map(class_mapping)

print("Class Mapping:", class_mapping)

########################################

# Data augmentation RANDOM

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

########################################

transformed_images = [(train_df.iloc[i]['image_path'], train_df.iloc[i]['class_name']) for i in random_indices]
print(original_images[:])

transformed_images_pil = [train_transform(Image.open(img[0]).convert("RGB")) for img in transformed_images]

# Create a 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        idx = i * 5 + j  # Image index
        axes[i, j].imshow(transformed_images_pil[idx].permute(1, 2, 0)) # Permute to (H, W, C)
        axes[i, j].set_title(transformed_images[idx][1])
        axes[i, j].axis("off")

plt.tight_layout()
# Save plot
plot_path_transformed = os.path.join(base_dir, "example_transformed_images.png")
plt.savefig(plot_path_transformed, bbox_inches="tight")

########################################

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

########################################

# Custom Dataset Class
class WasteDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        label = self.dataframe.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

########################################

# DataLoaders
train_dataset = WasteDataset(train_df, transform=train_transform)
val_dataset = WasteDataset(val_df, transform=val_test_transform)
test_dataset = WasteDataset(test_df, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

########################################

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=num_classes
    )

########################################

num_classes = len(class_mapping)
model.classifier = nn.Sequential(
    nn.Dropout(0.48699113095794067),
    nn.Linear(model.config.hidden_size, num_classes) # Use model.config.hidden_size to get input features for classifier
)

# # Load model checkpoint from "best_vit_75.pth"
# model.load_state_dict(torch.load("best_vit_75.pth"))

# Model -> GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# Optim and Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.0023280870643464266)
criterion = torch.nn.CrossEntropyLoss()

# Metrics
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
# Checkpoint
print("Train labels unique values:", train_df['label'].unique())
print("Val labels unique values:", val_df['class_name'].unique())
print("Test labels unique values:", test_df["class_name"].unique())



########################################

# Training (afegim checkpoints)
num_epochs = 75
best_val_acc = 0.0

# Reset metrics lists before training
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images, labels = images.to(device), labels.to(device).long()

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
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images).logits
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
            model_name = f"{experiment_name}-{datetime.now().strftime('%Y%m%d')}.pth"
            # Save model checkpoint in results_dir
            model_path = os.path.join(results_dir, model_name)
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with Accuracy: {val_accuracy * 100:.2f}%")

print(f"Training complete! Best model saved as: {model_name}" )

########################################

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
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
conf_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)