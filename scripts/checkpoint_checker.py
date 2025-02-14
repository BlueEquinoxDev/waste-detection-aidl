import torch
import matplotlib.pyplot as plt
import os

CHECKPOINT_DIR = "app/checkpoint/"
CHECKPOINT_PATHS = []
for (dirpath, dirnames, filenames) in os.walk(CHECKPOINT_DIR):
    CHECKPOINT_PATHS.extend(filenames)
    break

CHECKPOINT_PATHS = sorted(CHECKPOINT_PATHS)
print(CHECKPOINT_PATHS)
CHECKPOINT_PATHS.remove('.DS_Store')

train_loss = []
val_loss = []
for path in CHECKPOINT_PATHS:
    print(path)
    print(os.path.join(CHECKPOINT_DIR, path))
    
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, path), weights_only=False)
    train_loss.append(checkpoint['train_loss'])
    val_loss.append(checkpoint['val_loss'])

plt.plot(train_loss)
plt.plot(val_loss)
plt.show()