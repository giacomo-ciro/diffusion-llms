import json
import os

import torch
import numpy as np
import lightning as pl
from torch.utils.data import Dataset, DataLoader
import sys

class MemmapTokenDataset(Dataset):
    """
    Reads data directly from disk using np.memmap. 
    When accessed using __getitem__(self, idx), returns a batch of (X, y)
        X (tensor): a sequence of len context_length starting at idx
        y (tensor): the same sequence shifted by 1
    if strided_context == True, it only returns whole samples (that is, stride == context_length)
    """
    def __init__(
            self,
            memmap_path,
            context_length,
            is_diffusion_training: bool = False,
            eos_token_id: int = None,      
            pad_token_id: int = None,     
        ):

        self.data = np.memmap(memmap_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
        self.is_diffusion_training = is_diffusion_training
        self.stride = context_length if self.is_diffusion_training else 1
        
        # Calculate effective length - ensure we can always get context_length + 1 tokens
        # (for the shifted target sequence)
        total_positions = max(0, len(self.data) - (context_length + 1) + 1)
        self.effective_length = (total_positions + self.stride - 1) // self.stride
        
        # Setup for variable length generation
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
    def __len__(self):
        return self.effective_length
    
    def __getitem__(self, idx):
        if idx >= self.effective_length:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.effective_length} samples.")
        
        # Get the correct idx in the memmap object
        idx = idx * self.stride

        # Get sequence of indices
        # of shape (context_length,)
        X = self.data[idx:idx + self.context_length].copy()

        # Shifted by 1
        # of shape (context_length,)
        y = self.data[idx+1:idx + self.context_length+1].copy()
        
        # If autoregressive training, all the tokens are predicted (with masked attn)
        mask = torch.ones((X.shape[0],)).to(torch.bool)

        # If diffusion, compute the mask
        if self.is_diffusion_training:
            # Random mask for the output sequence
            # of shape # (context_length,)
            t = torch.rand(1).item()
            mask = torch.rand(
                size=(y.shape[0],), 
                dtype=torch.float32
            ) < t
            # TODO: always mask <eos>?

        # Cast to correct type
        X = torch.from_numpy(X).to(torch.int64)
        y = torch.from_numpy(y).to(torch.int64)

        # (int, int, bool)
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
        self.batch_size = self.config["batch_size"]
        self.val_test_perc = self.config["val_test_perc"]
        self.num_workers = 2
    
    def setup(self, stage=None):
        
        if os.path.exists(self.memmap_path):
            self.data = MemmapTokenDataset(
                self.memmap_path, 
                self.context_length,
                is_diffusion_training=self.config["pipeline"] == "diffusion",
                eos_token_id=self.config.get("eos_token_id", None),
                pad_token_id=self.config.get("pad_token_id", None)
            )
        else:
            print(f"[!] Can't find {self.memmap_path}, please create it using prepare.py")
            sys.exit()
        
        # Split the dataset
        assert self.val_test_perc < 1.0
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            self.data,
            [1.0 - 2*self.val_test_perc, self.val_test_perc, self.val_test_perc]
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