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
EPOCHS = 3
BATCH_SIZE = 10
LEARNING_RATE = 2e-5

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

print(f"Number of different labels in the train dataset: {train_dataset.idx_to_class} length: {len(train_dataset.idx_to_class)}")

# Define Model
model = ViTForImageClassification(len(train_dataset.idx_to_class))    
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
print("Detected Classes are: ", train_dataset.class_to_idx) 

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_loader  = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

# # Train the model
# for epoch in range(EPOCHS):        
#   for step, (x, y) in enumerate(train_loader):
#     # Change input array into list with each batch being one element
#     x = np.split(np.squeeze(np.array(x)), BATCH_SIZE)
#     # Remove unecessary dimension
#     for index, array in enumerate(x):
#       x[index] = np.squeeze(array)
#     # Apply feature extractor, stack back into 1 tensor and then convert to tensor
#     x = torch.tensor(np.stack(feature_extractor(x)['pixel_values'], axis=0))
#     # Send to GPU if available
#     x, y  = x.to(device), y.to(device)
#     b_x = Variable(x)   # batch x (image)
#     b_y = Variable(y)   # batch y (target)
#     # Feed through model
#     output, loss = model(b_x, None)
#     # Calculate loss
#     if loss is None: 
#       loss = loss_func(output, b_y)   
#       optimizer.zero_grad()           
#       loss.backward()                 
#       optimizer.step()

#     if step % 50 == 0:
#       # Get the next batch for testing purposes
#       test = next(iter(test_loader))
#       test_x = test[0]
#       # Reshape and get feature matrices as needed
#       test_x = np.split(np.squeeze(np.array(test_x)), BATCH_SIZE)
#       for index, array in enumerate(test_x):
#         test_x[index] = np.squeeze(array)
#       test_x = torch.tensor(np.stack(feature_extractor(test_x)['pixel_values'], axis=0))
#       # Send to appropirate computing device
#       test_x = test_x.to(device)
#       test_y = test[1].to(device)
#       # Get output (+ respective class) and compare to target
#       test_output, loss = model(test_x, test_y)
#       test_output = test_output.argmax(1)
#       # Calculate Accuracy
#       accuracy = (test_output == test_y).sum().item() / BATCH_SIZE
#       print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)

# Train the model
model.train()
for epoch in range(EPOCHS):        
    for step, batch in enumerate(train_loader):
        
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
        # print(f"loss: {loss}")
        # loss = loss_func(outputs.logits, y)
        
        if loss is None: 
              loss = loss_func(output, y)   
              optimizer.zero_grad()           
              loss.backward()                 
              optimizer.step()
        
        if step % 20 == 0:
            # print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step+1}], Loss: {loss:.4f}")
            # Get the next batch for testing purposes
            test = next(iter(test_loader))
            test_x = test['pixel_values'].to(device)
            test_y = test['labels'].to(device)

            model = model.eval()
            # Get output (+ respective class) and compare to target
            test_output, loss = model(test_x, test_y)
            preds = test_output.argmax(1)
            print(f"preds: {[train_dataset.idx_to_class[pred.item()] for pred in preds]}, test_y: {[test_dataset.idx_to_class[y.item()] for y in test_y]}")  

            # Print pictures with their predicted and actual labels
            for i in range(len(test_x)):
                plt.imshow(test_x[i].cpu().numpy().transpose(1, 2, 0))
                plt.title(f"Predicted: {train_dataset.idx_to_class[preds[i].item()]}, Actual: {test_dataset.idx_to_class[test_y[i].item()]}")
                plt.show()

            # Calculate Accuracy
            accuracy = (preds == test_y).sum().item() / BATCH_SIZE
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Step [{step+1}] | train loss: {loss:.4f} | test accuracy: {accuracy:.2f}")