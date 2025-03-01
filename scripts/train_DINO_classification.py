import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import timm  # Pretrained models, including DINOv2
import json

from custom_datasets.taco_viola_dataset_dino import TacoViolaDatasetDINO  # Updated dataset loader

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class mapping from JSON file
with open("data/taco5_categories.json", "r") as f:
    mapping_data = json.load(f)

# Convert taco5_categories.json structure to a dictionary {supercategory_id: cluster-category}
mapping_taco2viola = {}
classes = []

for category in mapping_data:
    cluster_name = category["cluster-category"]
    classes.append(cluster_name)  # Collect all class names
    for supercat_id in category["supercategories"]:
        mapping_taco2viola[supercat_id] = cluster_name  # Map each supercategory to its cluster

# Ensure that unclassified categories get a default value (e.g., "Unknown")
default_class = "Unknown"
mapping_taco2viola = {k: mapping_taco2viola.get(k, default_class) for k in range(60)}  

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.ToImage(),  # Convert to tensor
    transforms.ToDtype(torch.uint8, scale=True),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = TacoViolaDatasetDINO(
    annotations_file="data/train_annotations.json",
    img_dir="data/images",
    transform=transform,
    classes=classes,
    mapping=mapping_taco2viola
)

val_dataset = TacoViolaDatasetDINO(
    annotations_file="data/validation_annotations.json",
    img_dir="data/images",
    transform=transform,
    classes=classes,
    mapping=mapping_taco2viola
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

#DEBUG: UNIQUE LABELS
unique_labels = set()
for batch in train_loader:
    unique_labels.update(batch["labels"].tolist())

print(f"🚀 Unique labels found in dataset: {unique_labels}")

# Load pre-trained DINOv2 model
num_classes = len(set(unique_labels))  # Ensure it matches dataset labels
model = timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=num_classes)

print(f"✅ Adjusted num_classes: {num_classes}")

model = model.to(DEVICE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training function
def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct, total = 0, 0
        loop = tqdm(train_loader, leave=True)
        
        for batch in loop:
            images = batch["pixel_values"].to(DEVICE)
            labels = torch.tensor(batch["labels"], dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            loop.set_postfix(loss=running_loss / (total // BATCH_SIZE), acc=100 * correct / total)

        # Validation
        validate()

def validate():
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["pixel_values"].to(DEVICE)
            labels = torch.tensor(batch["labels"], dtype=torch.long).to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
