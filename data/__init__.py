from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
import os
import PIL

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            
    def __len__(self):
        return len(self.img_labels)

        
    def __getitem__(self, idx):
        image = PIL.Image.open(os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]))
        label = PIL.Image.open(os.path.join(self.img_dir, self.img_labels.iloc[idx, 1]))
        
        
        if self.transform:
            label = self.transform(label)
            image = self.transform(image)
          
        return image, label