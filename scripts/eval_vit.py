import torch
from PIL import Image
from datasets import load_dataset
from custom_datasets.viola77_dataset import Viola77Dataset
from model.waste_vit import WasteViT
import random

def eval_vit(model_path = "results/202502012134_ViT_Viola_TACO", image_path = None):
    
    # Create an instance of the model with the checkpoint
    model = WasteViT(checkpoint=model_path)


    ### Load an image ### 
    # Load the dataset
    dataset = load_dataset("viola77data/recycling-dataset", split="train")
    
    # Split dataset into training, validation, and test sets
    train_test = dataset.train_test_split(test_size=0.2)
    val_test = train_test["test"].train_test_split(test_size=0.5)
    test_dataset = val_test["test"]

    test_dataset = Viola77Dataset(test_dataset)

    # If no image_path provided then select a random image from the test dataset
    if image_path is None:
        idx = random.randint(0, len(test_dataset))
        # The dataset already contains the image data, no need to open it
        image = test_dataset.dataset[idx]['image']
    else:
        # If a path is provided, open it
        image = Image.open(image_path)

    # Apply transforms to the image to match the model's requirements
    inputs = model.feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class
    predicted_class_idx = logits.argmax(-1).item()
    print(f"Predicted class: {test_dataset.idx_to_cluster_class[predicted_class_idx]}")

eval_vit()