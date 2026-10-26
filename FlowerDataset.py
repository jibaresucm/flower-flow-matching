from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import numpy as np



class FlowersDataset(Dataset):
    
    def __init__(self, labels, image_size = 128):
        self.labels = labels 
        self.image_size = image_size

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        filename = f"image_{index + 1:05d}.jpg"
        filepath = os.path.join("./flowers", filename)

        img = Image.open(filepath).convert("RGB")
        img = img.resize((self.image_size, self.image_size))

        tens = torch.from_numpy(np.array(img).copy())
        tens = tens.permute(2, 0, 1).float().div_(255.0)
        tens = tens * 2.0 - 1.0

        label = torch.tensor(self.labels[index] - 1, dtype=torch.long)
        return tens, label