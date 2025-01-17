import argparse
import numpy as np
import os
import random
import torch

from dataset_classes.taco_dataset import TacoDataset
from utilities.config_utils import TaskType
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader

# Fix seed to be able to reproduce experiments
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Set device to be able to use GPU, MPS or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument("--annotations_dir", help="directory containing the data (Images)", type=str, default="./data")
parser.add_argument("--data_dir", help="directory containing the data (Images)", type=str, default="./data")
parser.add_argument("--task", help="either SEGMENTATION or CLASSIFICATION", type=str, default="SEGMENTATION")
parser.add_argument("--log_framework", help="either tensorboard or wandb", type=str, default="tensorboard")

parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--batch_size", help="batch size", type=int, default=20)
parser.add_argument("--n_epochs", help="training number of epochs", type=int, default=5)
parser.add_argument("--n_workers", help="number of workers for data loading", type=int, default=4)

parser.add_argument("--verbose", help="print info during execution", type=bool, default=True)

args = parser.parse_args()

# Check if the arguments are valid
assert os.path.exists(args.annotations_dir), "Annotations directory does NOT exist"
assert os.path.exists(args.data_dir), "Data directory does NOT exist"
assert args.task in ['SEGMENTATION', 'CLASSIFICATION'], "Task NOT valid. The options are either SEGMENTATION or CLASSIFICATION"
assert args.log_framework in ['tensorboard', 'wandb'], "Framework NOT valid. The options are either tensorboard or wandb"

# Generate paths to the annotations files
train_annotations_file = os.path.join(args.annotations_dir, "train_annotations.json")
val_annotations_file = os.path.join(args.annotations_dir, "validation_annotations.json")
test_annotations_file = os.path.join(args.annotations_dir, "test_annotations.json")

# Create the transforms for training, validation and testing
# TODO: Compute real mean and std of channels + Add more data augmentation techniques
#data_transforms_train = transforms.Compose([
    #transforms.ToImage(),  # To tensor is deprecated
    #transforms.ToDtype(torch.uint8, scale=True),
    #transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
    #transforms.RandomRotation(degrees=15),
    #transforms.RandomHorizontalFlip(0.5),
    #transforms.ToDtype(torch.float32, scale=True),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#    ])

data_transforms_test = transforms.Compose([
    transforms.ToImage(),  # To tensor is deprecated,
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Create the datasets for training, validation and testing
train_dataset = TacoDataset(annotations_file=train_annotations_file, img_dir=args.data_dir, task=TaskType[args.task.upper()])
val_dataset = TacoDataset(annotations_file=val_annotations_file, img_dir=args.data_dir, transforms=data_transforms_test, task=TaskType[args.task.upper()])
test_dataset = TacoDataset(annotations_file=test_annotations_file, img_dir=args.data_dir, transforms=data_transforms_test, task=TaskType[args.task.upper()])

# Print the number of images in each dataset
if args.verbose: 
    print(f"Number of images in the training set: {len(train_dataset)}")
    print(f"Number of images in the validation set: {len(val_dataset)}")
    print(f"Number of images in the testing set: {len(test_dataset)}")

# Create the dataloaders for training, validation and testing
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)



