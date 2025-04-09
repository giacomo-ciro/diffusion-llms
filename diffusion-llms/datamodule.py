import json
import os

import torch
import numpy as np
import lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

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
            stride=None
        ):

        self.data = np.memmap(memmap_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
        self.stride = stride if stride is not None else 1
        
        # Calculate effective length - ensure we can always get context_length + 1 tokens
        # (for the shifted target sequence)
        self.effective_length = max(0, (len(self.data) - (context_length + 1)) // self.stride + 1)
        
    def __len__(self):
        return self.effective_length
    
    def __getitem__(self, idx):
        if idx >= self.effective_length:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.effective_length} samples")

        # Calculate start position based on stride
        start_idx = idx * self.stride
        
        # Get sequence of indices
        X = self.data[start_idx:start_idx + self.context_length].copy()

        # Shifted by 1
        y = self.data[start_idx+1:start_idx + self.context_length+1].copy()
        
        # Random mask
        # mask = torch.rand(
        #     size=(sequence.shape[0],), 
        #     dtype=torch.float32
        # ) < 0.15
        # Both are (context_length,)
        return torch.from_numpy(X), torch.from_numpy(y)
    
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
            self.context_length
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