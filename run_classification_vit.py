from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
from utilities.config_utils import TaskType
from datasets.taco_dataset_vit import TacoDatasetViT
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from torchvision.transforms import v2 as transforms
import os
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the feature extractor and model
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

train_annotations_file = os.path.join("data", "train_annotations.json")
val_annotations_file = os.path.join("data", "validation_annotations.json")
test_annotations_file = os.path.join("data", "test_annotations.json")

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

print(f"Train dataset length: {len(train_dataset)}")
print(f"size of train_dataset[0]: {train_dataset[0].__sizeof__}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")  

# TEST IMAGE SIZE Get the first item from the dataset
#sample = train_dataset[0]

# TEST IMAGE SIZE Assuming the first element of the tuple is the image tensor
#image_tensor = sample['pixel_values']  # Adjust the index based on your dataset's structure
#labels = sample['labels']

# TEST IMAGE SIZE Convert the tensor to a PIL image
#image = transforms.ToPILImage()(image_tensor)

# TEST IMAGE SIZE Print the size of the image
#print(f"Image size: {image.size}")

# Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8)
# test_loader = DataLoader(test_dataset, batch_size=8)

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

def compute_metrics(eval_pred):
    """Compute metrics for classification task"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Detailed classification report
    report = classification_report(
        labels, 
        predictions, 
        target_names=list(train_dataset.idx_to_cluster_class.values()),
        output_dict=True
    )
    
    metrics = {
        'accuracy': accuracy,
    }
    
    # Add per-class metrics
    for class_name, class_metrics in report.items():
        if isinstance(class_metrics, dict):
            metrics[f'{class_name}_precision'] = class_metrics['precision']
            metrics[f'{class_name}_recall'] = class_metrics['recall']
            metrics[f'{class_name}_f1'] = class_metrics['f1-score']
    
    return metrics

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    processing_class=feature_extractor,  # Changed from tokenizer to processing_class
)

# Train the model
trainer.train()

# Evaluate the model
val_results = trainer.evaluate()
print(f'Evaluation results: {val_results}')