import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MoviePosterDataset(Dataset):
    def __init__(self, parquet_path, img_dir, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.img_dir = img_dir
        self.transform = transform
        # Isolate only your 19 genre binary classification columns
        self.genre_cols = [col for col in self.df.columns if col != 'id']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{int(row['id'])}.jpg")
        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[self.genre_cols].values.astype('float32'))
        
        if self.transform:
            image = self.transform(image)
        return image, labels

