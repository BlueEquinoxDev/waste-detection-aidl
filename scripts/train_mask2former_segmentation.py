from torch.utils.data import DataLoader
from transformers import MaskFormerImageProcessor
import numpy as np
from custom_datasets.taco_dataset_mask2former import TacoDatasetMask2Former 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import MaskFormerForInstanceSegmentation
import torch
from tqdm.auto import tqdm

processor = MaskFormerImageProcessor(
    reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False
    # do_resize=True,
    # size={"height": 512, "width": 512},
    # do_normalize=False
)


# Create transform pipeline that handles both image and mask
transform = A.Compose([
    A.Resize(height=512, width=512),
    ToTensorV2()
])

# Initialize dataset with transforms
train_dataset = TacoDatasetMask2Former(
    annotations_file="data/train_annotations.json",
    img_dir="data/images",
    processor=processor,
    transforms=transform
)
# print(f"len(dataset): {len(train_dataset)}")
# print(f"dataset[0]: {train_dataset[1]}")

# for k, v in train_dataset[1].items():
#     print(f"{k} {v.shape}")
#     print(f"{k}: {v}")
#     print(f"unique values: {np.unique(v)}")

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"].float() / 255.0 for example in batch])  # Convert to float and normalize
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

batch = next(iter(train_dataloader))
# for k,v in batch.items():
#   if isinstance(v, torch.Tensor):
#     #print(k,v.shape)
#   else:
# #    print(k,len(v))



# We specify ignore_mismatched_sizes=True to replace the already fine-tuned classification head by a new one
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=train_dataloader.dataset.idx2class,
                                                          ignore_mismatched_sizes=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0
for epoch in range(100):
    print("Epoch:", epoch)
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size

        if idx % 100 == 0:
            print("Loss:", running_loss/num_samples)

        # Optimization
        optimizer.step()