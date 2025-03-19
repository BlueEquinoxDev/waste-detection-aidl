import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from datasets import load_dataset
from custom_datasets.viola_dataset_resnet import Viola77DatasetResNet
# from scripts.train_resnet_classification import test_dataset

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = load_dataset("viola77data/recycling-dataset", split="train")
SEED = 42
# Split dataset into training, validation, and test sets
train_test = dataset.train_test_split(test_size=0.2, seed=SEED)
val_test = train_test["test"].train_test_split(test_size=0.5, seed=SEED)

test_dataset = val_test["test"]
test_dataset = Viola77DatasetResNet(test_dataset, transform=val_test_transform)

# Define DataLoader (num_workers=0 to avoid multiprocessing error on Windows)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
print(test_dataset[0])
print(f"Dataset length: {len(test_dataset)}")
sample = test_dataset[0]  # Get first sample

# Print test_dataset characteristics:

print(f"Sample keys: {sample.keys()}")  # Print keys (should be 'pixel_values' and 'labels')

if isinstance(sample["pixel_values"], torch.Tensor):
    print(f"Sample image shape: {sample['pixel_values'].shape}")  # Should be (C, H, W)
    print(f"Sample label: {sample['labels']} (Type: {type(sample['labels'])})")
else:
    print(f"Sample image type: {type(sample['pixel_values'])}")
    print(f"Sample label type: {type(sample['labels'])}")

def load_model():
    model_path = "./model/best_resnet50.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize ResNet-50
    model = models.resnet50(pretrained=False)

    # Ensure the classifier matches the saved model (fc.1.weight is present)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, len(test_dataset.classes))
    # )
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Dropout(0.2674),  # Match dropout if used in training
    nn.Linear(num_ftrs, len(test_dataset.classes))  # Ensure correct number of output classes
    )

    model = model.to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, device

def evaluate_model():
    model, device = load_model()
    true_labels, pred_labels, probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['pixel_values']
            labels = batch['labels']
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            probs.extend(probabilities.cpu().numpy())


    results_dir = "./metrics/resnet/test"
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Save Overall Accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    accuracy_file = os.path.join(results_dir, "accuracy.txt")
    with open(accuracy_file, "w") as f:
        f.write(f'Overall Accuracy: {accuracy:.4f}\n')
    print(f'Overall Accuracy: {accuracy:.4f} (saved to {accuracy_file})')

    # Save Classification Report
    classification_report_str = classification_report(true_labels, pred_labels, target_names=test_dataset.classes)
    report_file = os.path.join(results_dir, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(classification_report_str)
    print(f'Classification Report saved to {report_file}')

    # Save Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    conf_matrix_file = os.path.join(results_dir, "TEST_confusion_matrix.png")
    plt.savefig(conf_matrix_file)
    plt.show()
    plt.close()  # Close figure to free memory
    print(f'Confusion Matrix saved to {conf_matrix_file}')

    # Save Histogram of Per-Class Accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(test_dataset.classes)), class_accuracies, tick_label=test_dataset.classes)
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy Distribution")
    hist_file = os.path.join(results_dir, "TEST_hist.png")
    plt.savefig(hist_file)
    plt.show()
    plt.close()  # Close figure to free memory
    print(f'Per-Class Accuracy Histogram saved to {hist_file}')


if __name__ == "__main__":
    evaluate_model()
