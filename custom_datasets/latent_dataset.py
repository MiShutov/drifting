import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import numpy as np


class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.latent_dir = Path(latent_dir)
        self.latent_files = sorted(list(self.latent_dir.glob("*.pt")))
        print(f"Number of latent samples: {len(self.latent_files)}")
        
    def __len__(self):
        return len(self.latent_files)
    
    def __getitem__(self, idx):
        latent = torch.load(self.latent_files[idx])
        return latent.to(torch.bfloat16)