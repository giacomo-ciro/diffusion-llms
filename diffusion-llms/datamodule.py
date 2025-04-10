import json
import os

import torch
import numpy as np
import lightning as pl
from torch.utils.data import Dataset, DataLoader

class MemmapTokenDataset(Dataset):
    """
    Reads data directly from disk using np.memmap. 
    When accessed using __getitem__(self, idx), returns a batch of (X, y)
        X (tensor): a sequence of len context_length starting at idx
        y (tensor): the same sequence shifted by 1
    """
    def __init__(
            self,
            memmap_path,
            context_length,
            is_diffusion_training: bool = False,
        ):

        self.data = np.memmap(memmap_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
        self.is_diffusion_training = is_diffusion_training
        
        # Calculate effective length - ensure we can always get context_length + 1 tokens
        # (for the shifted target sequence)
        self.effective_length = max(0, (len(self.data) - (context_length + 1)) + 1)
        
    def __len__(self):
        return self.effective_length
    
    def __getitem__(self, idx):
        if idx >= self.effective_length:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.effective_length} samples")
        
        # Get sequence of indices
        # of shape (context_length,)
        X = self.data[idx:idx + self.context_length].copy()

        # Shifted by 1
        # of shape (context_length,)
        y = self.data[idx+1:idx + self.context_length+1].copy()
        
        # If autoregressive training, return X,y
        if not self.is_diffusion_training:
            return torch.from_numpy(X), torch.from_numpy(y)
        
        # Random mask for the output sequence
        # of shape # (context_length,)
        t = torch.rand(1).item()
        mask = torch.rand(
            size=(y.shape[0],), 
            dtype=torch.float32
        ) < t

        return X, y, mask
        
        
    
class MemmapDataModule(pl.LightningDataModule):
    """
    The main datamodule. When iterated over, returns batches of (X, y) of sequence and target sequence shifted by one.
    """
    def __init__(
        self, 
        config_path
    ):
        super().__init__()

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.memmap_path = self.config["memmap_path"]
        self.context_length = self.config["context_length"]
        self.mask_ratio = self.config["mask_ratio"]
        self.batch_size = self.config["batch_size"]
        self.val_test_tokens = self.config["val_test_tokens"]
        self.num_workers = 2
    
    def setup(self, stage=None):
        
        if os.path.exists(self.memmap_path):
            self.data = MemmapTokenDataset(
            self.memmap_path, 
            self.context_length,
            is_diffusion_training=self.config["pipeline"] == "diffusion"
        )
        else:
            print(f"[!] Can't find {self.memmap_path}, please create it using prepare.py")
            exit()
        
        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            self.data,
            # Use the given number for test and val, rest for token
            [len(self.data) - 2*self.val_test_tokens, self.val_test_tokens, self.val_test_tokens]
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )