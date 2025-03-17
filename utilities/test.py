import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

# Define three example masks
mask1 = np.array([[0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]])

mask2 = np.array([[0, 0, 2],
                  [0, 2, 2],
                  [0, 0, 0]])

mask3 = np.array([[3, 0, 0],
                  [3, 0, 0],
                  [0, 0, 0]])

masks = [mask1, mask2, mask3]
print(f"masks shape: {masks[0].shape}")

# Apply masking to each mask so that zeros are hidden
masked_masks = [ma.masked_where(mask == 0, mask) for mask in masks]

# Plot the original and masked versions of each mask
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, masked in enumerate(masked_masks):
    axes[i].imshow(masked, cmap='BrBG_r', interpolation='none')
    axes[i].set_title(f'Mask {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
