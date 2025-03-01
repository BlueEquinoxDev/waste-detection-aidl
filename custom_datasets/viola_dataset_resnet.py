from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class Viola77DatasetResNet(Dataset):
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset: A Hugging Face dataset containing at least 'image' and 'label' fields.
            transform: Optional transformations to be applied on the image.
        """
        self.dataset = dataset
        # If no transform is provided, default to converting image to tensor.
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        # Print and store available class names from the dataset features
        print(f"Available classes: {self.dataset.features['label'].names}")
        self.classes = self.dataset.features['label'].names

        # Create a dictionary mapping class names to indices
        self.cluster_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        # Create the reverse mapping from indices to class names
        self.idx_to_cluster_class = {idx: cls_name for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']
    

        # If the image is a string (e.g., a file path), open it as a PIL Image.
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                raise ValueError(f"Error opening image at index {idx}: {e}")

        # If the image is not already a PIL Image, attempt to convert if it’s a numpy array.
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception as e:
                raise ValueError(f"Could not convert image at index {idx} to PIL Image: {e}")

        # Apply the provided transform to convert the PIL Image to a tensor.
        image = self.transform(image)

        return {'pixel_values': image, 'labels': label}

