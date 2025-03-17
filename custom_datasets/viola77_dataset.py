from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt


class Viola77Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        # Print available classes in the dataset
        print(f"Available classes: {self.dataset.features['label'].names}")
        self.classes = self.dataset.features['label'].names

        # Create a dictionary to map the class names to indices
        self.cluster_class_to_idx = { cls_name: idx for idx, cls_name in enumerate(self.classes) }
        # print(f"Cluster class to idx: {self.cluster_class_to_idx}")

        # Reverse mapping
        self.idx_to_cluster_class = {idx: cls_name for idx, cls_name in enumerate(self.cluster_class_to_idx.keys())}
        # print(f"Idx to cluster class: {self.idx_to_cluster_class}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        
        # Apply transforms if they exist
        if self.transform:
            image = self.transform(image)

        return {'pixel_values': image, 'labels': label}