from transformers import ViTFeatureExtractor
import torch.nn as nn
import torch
from model.vit_for_image_classification import ViTForImageClassification
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
from datasets.taco_dataset_vit import TacoDatasetViT
from torchvision.transforms import v2 as transforms
import os
import matplotlib.pyplot as plt

# Define Constants
EPOCHS = 10
BATCH_SIZE = 10
TEST_BATCH_SIZE = 25
LEARNING_RATE = 2e-5

def main():
    # Define the dataset
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

    # Load the TACO dataset
    train_dataset = TacoDatasetViT(annotations_file=train_annotations_file, img_dir="data", transforms=data_transforms_train)
    val_dataset = TacoDatasetViT(annotations_file=val_annotations_file, img_dir="data", transforms=data_transforms_test)
    test_dataset = TacoDatasetViT(annotations_file=test_annotations_file, img_dir="data", transforms=data_transforms_test)

    print(f"Number of different labels in the train dataset: {train_dataset.idx_to_cluster_class} length: {len(train_dataset.idx_to_cluster_class)}")

    # Define Model
    model = ViTForImageClassification(len(train_dataset.idx_to_cluster_class))    
    # Feature Extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cross Entropy Loss
    loss_func = nn.CrossEntropyLoss()
    # Use GPU if available  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if torch.cuda.is_available():
        model.cuda() 

    print("Number of train samples: ", len(train_dataset))
    print("Number of test samples: ", len(test_dataset))
    print("Detected Classes are: ", train_dataset.cluster_class_to_idx) 

    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_loader  = data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=4) 

    # Train the model
    for epoch in range(EPOCHS):        
        for step, batch in enumerate(train_loader):
            model.train()        
            # print(f"Step: {step}, batch: {batch}, batch type: {type(batch)}")  

            # Extract pixel values and labels from the batch dictionary
            x = batch['pixel_values']  # Should already be a tensor
            y = batch['labels']
            
            # print(f"Step: {step}, x shape: {x.shape}, y shape: {y.shape}")
            
            # Send to GPU if available
            x, y = x.to(device), y.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, loss = model(x, y)
            
            if loss is None: 
                loss = loss_func(output, y)   
                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step()
            
            if step % 74 == 0:
                # Get the next batch for testing purposes
                test = next(iter(test_loader))
                test_x = test['pixel_values'].to(device)
                test_y = test['labels'].to(device)

                model = model.eval()
                # Get output (+ respective class) and compare to target
                test_output, loss = model(test_x, test_y)
                preds = test_output.argmax(1)

                # Print predictions and actual labels
                print(f"preds: {[train_dataset.idx_to_cluster_class[pred.item()] for pred in preds]}, test_y: {[test_dataset.idx_to_cluster_class[y.item()] for y in test_y]}")  

                # Print pictures with their predicted and actual labels
                # for i in range(len(test_x)):
                #     plt.imshow(test_x[i].cpu().numpy().transpose(1, 2, 0))
                #     plt.title(f"Predicted: {train_dataset.idx_to_cluster_class[preds[i].item()]}, Actual: {test_dataset.idx_to_cluster_class[test_y[i].item()]}")
                #     plt.show()

                # Calculate Accuracy
                accuracy = (preds == test_y).sum().item() / TEST_BATCH_SIZE
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Step [{step+1}] | train loss: {loss:.4f} | test accuracy: {accuracy:.2f}")
    

if __name__ == '__main__':
    main()