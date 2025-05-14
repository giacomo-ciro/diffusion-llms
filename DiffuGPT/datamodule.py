import json
import os

import torch
import numpy as np
import lightning as pl
from torch.utils.data import Dataset, DataLoader
import sys

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokenDataset(Dataset):
    def __init__(self, config: dict):
        super().__init__()
        # Load raw data
        self.data = np.memmap(config["memmap_path"], dtype=np.uint16, mode='r')
        self.context_length = config["context_length"]
        self.eos_token_id = config["eos_token_id"]
        self.pad_token_id = config["pad_token_id"]
        # If you ever want to use a special [MASK] token, set it here. By default, we reuse EOS.
        self.mask_token_id = config.get("mask_token_id", self.eos_token_id)

        # Sliding-window parameters
        self.stride = self.context_length
        total_positions = max(0, len(self.data) - (self.context_length + 1) + 1)
        self.effective_length = (total_positions + self.stride - 1) // self.stride

        # Probability of *choosing* the random-masking logic
        self.random_mask_prob = config.get("random_mask_prob", 0.1)
        # Max window size around EOS (will be annealed from 0 → this)
        self.eos_window_max = min(config.get("eos_window_max", 100), 100)
        # Number of steps over which to linearly grow the window
        self.window_annealing_steps = config.get("window_annealing_steps", 5000)
        if self.window_annealing_steps > 0:
            self.window_schedule = np.linspace(
                0,
                self.eos_window_max,
                self.window_annealing_steps,
                dtype=int
            )
        else:
            # If no annealing, always use full window
            self.window_schedule = np.array([self.eos_window_max], dtype=int)

    def __len__(self):
        return self.effective_length

    def __getitem__(self, idx):
        # 1) load a sliding window of length context_length+1
        start = idx * self.stride
        seq = self.data[start : start + self.context_length + 1].copy()
        X_np = seq[: self.context_length]
        y_np = seq[1 : self.context_length + 1]
        # Ensure the last target is EOS
        y_np[-1] = self.eos_token_id
        # Padded part

        # 2) to torch
        X = torch.from_numpy(X_np).long()
        y = torch.from_numpy(y_np).long()

        return X, y
    
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
        self.batch_size = self.config["batch_size"]
        self.val_test_perc = self.config["val_test_perc"]
        self.num_workers = 2
    
    def setup(self, stage=None):
        
        if os.path.exists(self.memmap_path):
            self.data = MemmapTokenDataset(
                self.config
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