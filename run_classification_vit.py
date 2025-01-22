from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
from datasets.taco_dataset import TacoDataset, TaskType
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from torchvision.transforms import v2 as transforms
import os

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the TACO dataset
train_dataset = TacoDataset(annotations_file=train_annotations_file, img_dir="data", transforms=data_transforms_train, task=TaskType['CLASSIFICATION'])
val_dataset = TacoDataset(annotations_file=val_annotations_file, img_dir="data", transforms=data_transforms_test, task=TaskType['CLASSIFICATION'])
test_dataset = TacoDataset(annotations_file=test_annotations_file, img_dir="data", transforms=data_transforms_test, task=TaskType['CLASSIFICATION'])

print(f"Train dataset length: {len(train_dataset)}")
print(f"size of train_dataset[0]: {train_dataset[0].__sizeof__}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")  

# TEST IMAGE SIZE Get the first item from the dataset
sample = train_dataset[0]

# TEST IMAGE SIZE Assuming the first element of the tuple is the image tensor
image_tensor = sample[0]  # Adjust the index based on your dataset's structure

# TEST IMAGE SIZE Convert the tensor to a PIL image
image = transforms.ToPILImage()(image_tensor)

# TEST IMAGE SIZE Print the size of the image
print(f"Image size: {image.size}")

# Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8)
# test_loader = DataLoader(test_dataset, batch_size=8)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
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
    tokenizer=feature_extractor,
)

# Train the model
trainer.train()

# Evaluate the model
val_results = trainer.evaluate()
print(f'Evaluation results: {val_results}')