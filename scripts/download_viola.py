import os
import csv
import requests
from io import BytesIO
from PIL import Image
import pandas as pd
from collections import Counter
from datasets import load_dataset

import matplotlib.pyplot as plt

# Define constants
BASE_DIR = "data/viola"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations.csv")
UPDATED_ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations_with_classes.csv")
DATASET_NAME = "wastevisum/viola"
CLASS_MAPPING = {
    0: "recyclable",
    1: "non-recyclable"
}

def download_and_save_dataset():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, split="train")
    with open(ANNOTATIONS_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])
        for idx, data in enumerate(dataset):
            if "image" not in data or "label" not in data:
                print(f"Skipping entry {idx} due to missing keys.")
                continue

            image_data = data["image"]
            label = data["label"]

            # If image is a URL, download it; otherwise assume it's a PIL Image
            if isinstance(image_data, str):
                response = requests.get(image_data)
                image = Image.open(BytesIO(response.content))
            else:
                image = image_data

            image_path = os.path.join(IMAGES_DIR, f"image_{idx}.jpg")
            image.save(image_path)
            writer.writerow([image_path, label])

    print(f"Images saved in: {IMAGES_DIR}")
    print(f"Annotations saved in: {ANNOTATIONS_FILE}")

def update_annotations():
    df = pd.read_csv(ANNOTATIONS_FILE)
    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV file must contain 'image_path' and 'label' columns.")

    df["label"] = df["label"].astype(int)
    df["class_name"] = df["label"].map(CLASS_MAPPING)
    df.to_csv(UPDATED_ANNOTATIONS_FILE, index=False)
    print(f"Updated annotations saved to: {UPDATED_ANNOTATIONS_FILE}")
    return df

def plot_class_distribution(df):
    class_counts = Counter(df["class_name"])
    df_counts = pd.DataFrame(class_counts.items(), columns=["Class", "Count"]).sort_values(by="Count", ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_counts["Class"], df_counts["Count"], color="skyblue")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plot_path = os.path.join(BASE_DIR, "class_distribution.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    print(f"Class distribution plot saved to: {plot_path}")
