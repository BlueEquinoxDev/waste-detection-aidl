import os
from datetime import datetime
import torch

# Save checkpoint function
def save_model(model, epoch, optimizer, idx2class, results_dir):
    checkpoint = {
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "idx2classes": idx2class
    }  

    save_path = results_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # using now() to get current time
    current_time = datetime.now()
    filename = f"{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}"

    torch.save(checkpoint, os.path.join(save_path, f"checkpoint_epoch_{epoch}_{filename}.pt"))
