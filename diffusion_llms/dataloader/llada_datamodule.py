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

import numpy as np
import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
import numpy as np

class RegressionDataset(Dataset):
    def __init__(self, config: dict):
        super().__init__()
        
        # Open cvs with prompts
        self.prompts
        self.answers

        # Load raw data
        self.length = len(self.prompts)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        # Get elements at idx
        prompt = self.prompts[idx]
        ans = self.answers[idx]

        # Concat & create mask
        X = concat(prompt, ans)
        y = len(ans)
        msk = torch.zeros_like(X)
        # add ones at answer + eos positions
        
        return X, y, msk

class ClassificationDataset(Dataset):
    def __init__(self, config: dict):
        super().__init__()
        
        # Open cvs with prompts
        self.prompts
        self.answers

        # Load raw data
        self.length = len(self.prompts)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        # Get elements at idx
        prompt = self.prompts[idx]
        ans = self.answers[idx]

        # Concat & create mask
        X = concat(prompt, ans)
        y = # 0,1 is eos or not
        msk = torch.zeros_like(X)
        # add ones at answer + eos positions
        
        return X, y, msk
    
class DataModule(pl.LightningDataModule):
    """
    The main datamodule. When iterated over, returns batches of (X, y, msk) of sequence and target sequence shifted by one.
    """
    def __init__(
        self, 
        config_path
    ):
        super().__init__()

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.batch_size = self.config["batch_size"]
        self.val_test_perc = self.config["val_test_perc"]
        self.num_workers = 2
    
    def setup(self, stage=None):
        
        if task == "classification":
            # when iterated over, returns (X, y, msk) of shape
            # X = [B, context_length]
            # y = [B, context_length]
            # msk = [B, context_length]     (1 if belongs to answer + eos, 0 if belongs to prompt)
            self.data = ClassificationDataset(
                self.config
            )
            
        elif task == "regression":
            # when iterated over, returns (X, y, msk) of shape
            # X = [B, context_length]   
            # y = [B, 1]                
            # msk = [B, context_length]     (1 if belongs to answer + eos, 0 if belongs to prompt)
            self.data = RegressionDataset(
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