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

if '.DS_Store' in CHECKPOINT_PATHS:
    CHECKPOINT_PATHS.remove('.DS_Store')

train_loss = []
val_loss = []
for path in CHECKPOINT_PATHS:
    print(path)
    print(os.path.join(CHECKPOINT_DIR, path))
    
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, path), weights_only=False, map_location=torch.device('cpu'))
    train_loss.append(checkpoint['train_loss'])
    val_loss.append(checkpoint['val_loss'])


plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()