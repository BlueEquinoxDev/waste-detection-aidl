from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self):
        self.labels_map={
            0: 'T-shirt',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle Boot'
        }
    
    def labels_map(self):
        return self.labels_map
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class TraintMNIST(MNISTDataset):

    def __init__(self):
        super().__init__()
        self.train_data= datasets.FashionMNIST(root='data',
                                               train=True, 
                                               download=True, 
                                               transform=ToTensor(),)


    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, idx):
        return self.train_data[idx]


class ValidationMNIST(MNISTDataset):

    def __init__(self):
        super().__init__()
        self.train_data= datasets.FashionMNIST(root='data',
                                               train=False ,
                                               download=True, 
                                               transform=ToTensor(),)


    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, idx):
        return self.train_data[idx]

