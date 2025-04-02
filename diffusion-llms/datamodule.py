
import json
import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset 
import lightning as pl
import tiktoken

CONFIG_PATH = "./config.json"

class MemoryMappedDataset(Dataset):
    """
    Creates a memory-backed dataset to iterate over.
    First check whether the data.bin numpy array already exists, then read it. If not existent, create it.
    """
    def __init__(self, filename, dtype, shape, transform=None):
        self.memmap = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
        self.transform = transform
        
    def __len__(self):
        return len(self.memmap)
    
    def __getitem__(self, idx):
        # Get numpy array from memmap
        sample = self.memmap[idx].copy()  # Copy to avoid issues with the memmap
        
        # Convert to torch tensor
        sample = torch.from_numpy(sample)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    

class DataModule(pl.LightningDataModule):
    """
    Datamodule takes care of reading from disk the data (text), and returning a batch = (ids, targets).
    Data is read as a np.memmap() array to avoid loading everything on RAM
    """
    def __init__(self, dataset_path, size):
        super().__init__()
        with open(CONFIG_PATH, "r") as f:
            self.config = json.load(f)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.data = load_dataset(dataset_path, streaming=True)["train"].take(size)

    def prepare_data(self):
        # TODO: add code to download census SOMA here
        pass

    def setup(self, stage=None):
        

    def train_dataloader(self) -> DataLoader:
        return soma_ml.experiment_dataloader(self.train_dataset)
    
    def val_dataloader(self) -> DataLoader:
        return soma_ml.experiment_dataloader(self.val_dataset)
    
    def test_dataloader(self) -> DataLoader:
        return soma_ml.experiment_dataloader(self.test_dataset)