import torch
from PIL import Image
from datasets import load_dataset
import torchvision.transforms as transforms
from custom_datasets.viola77_dataset import Viola77Dataset
from model.waste_vit import WasteViT
import random
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

SEED = 42

parser = argparse.ArgumentParser(description='Select the model and (optional) an image for evaluation')
# get parameter model_path and image_path from the command line
parser.add_argument('--model_path', required=True, help='Dataset name', type=str)
parser.add_argument('--image_path', required=False, help='Dataset name', type=str, default=None)

def generate_evaluation_summary(accuracy, accuracy_per_class, f1, f1_per_class, cm, label_names):
    """
    Generates a figure with a table displaying overall accuracy and per-class metrics,
    along with a confusion matrix heatmap.
    
    Parameters:
        accuracy: Overall accuracy as a float
        accuracy_per_class: List of accuracies for each class
        f1: Overall F1 score
        f1_per_class: List of F1 scores for each class
        cm: Confusion matrix
        label_names: List of human-readable class names
    """
    # Create a figure with two subplots: one for the table and one for the heatmap.
    fig, (ax_table, ax_cm) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
    # Prepare table data with three columns
    table_data = [["Class", "Accuracy (%)", "F1 Score"]]
    
    # Add overall metrics as a special first row
    table_data.append(["Overall", f"{accuracy * 100:.2f}%", f"{f1:.4f}"])
    
    # Add per-class metrics with actual class names
    for idx in range(len(accuracy_per_class)):
        table_data.append([
            label_names[idx], 
            f"{accuracy_per_class[idx] * 100:.2f}%", 
            f"{f1_per_class[idx]:.4f}"
        ])
    
    # Hide the axes for the table
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create the table on ax_table
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # Adjust the row height
    
    # Plot the confusion matrix as a heatmap on ax_cm with class names
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", 
                xticklabels=label_names, yticklabels=label_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    
    plt.tight_layout()
    return fig

def eval_vit(model_path = None, image_path = None):
    
    results_path = "/".join(model_path.split("/")[:2])
    os.makedirs(results_path, exist_ok=True)
    print(f"Results will be saved in: {results_path}")

    # Define the transformations
    data_transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = load_dataset("viola77data/recycling-dataset", split="train")
    
    # Split dataset into training, validation, and test sets
    train_test = dataset.train_test_split(test_size=0.2, seed=SEED)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=SEED)
    test_dataset = val_test["test"]

    test_dataset = Viola77Dataset(test_dataset, transform=data_transforms_test)
    
    num_classes = len(test_dataset.idx_to_cluster_class)
    label_names = list(test_dataset.idx_to_cluster_class.values())
    id2label = {idx: label for idx, label in test_dataset.idx_to_cluster_class.items()}
    label2id = test_dataset.cluster_class_to_idx

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the model with the checkpoint
    model = WasteViT(num_classes=num_classes, id2label=id2label, label2id=label2id, checkpoint=model_path)
    model.to(device)

    # If an image_path is provided run inference for a single image
    if image_path is not None:
        image = Image.open(image_path)
        
        if isinstance(images[0], torch.Tensor):
            # Images are already tensors from your dataset transforms
            images_tensor = torch.stack(images).to(device)
            inputs = {'pixel_values': images_tensor}
        else:
            # Fall back to feature extractor if needed
            inputs = model.feature_extractor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(f"Predicted class: {test_dataset.idx_to_cluster_class[predicted_class_idx]}")
    else:
        # Evaluate the whole test dataset using
        def collate_fn(batch):
            images, labels = zip(*batch)
            return list(images), torch.tensor(labels)
        
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
        model.eval()
        all_preds = []
        all_labels = []
        
        for images, labels in test_loader:
            print(".", end="")
            
            if isinstance(images[0], torch.Tensor):
                # Images are already tensors from your dataset transforms
                images_tensor = torch.stack(images).to(device)
                inputs = {'pixel_values': images_tensor}
            else:
                # Fall back to feature extractor if needed
                inputs = model.feature_extractor(images=images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(all_labels, all_preds)
        total_correct = accuracy_score(all_labels, all_preds, normalize=False)  # Just for reference
        f1 = f1_score(all_labels, all_preds, average='weighted')
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        cm = confusion_matrix(all_labels, all_preds)

        # Calculate per-class accuracy properly
        accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

        print("Test Accuracy: {:.2f}%".format(accuracy * 100))
        print("Per-class Accuracy:", accuracy_per_class)
        print("F1 Score: {:.4f}".format(f1))
        print("F1 Score per class:", f1_per_class)
        print("Confusion Matrix:\n", cm)

        fig = generate_evaluation_summary(accuracy, accuracy_per_class, f1, f1_per_class, cm, label_names)
        results_file_name = os.path.join(results_path, "evaluation_summary.png")
        fig.savefig(results_file_name)


eval_vit(parser.parse_args().model_path, parser.parse_args().image_path)