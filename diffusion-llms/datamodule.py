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
            is_padded_dataset: bool = False,
            eos_token_id: int = None,      
            pad_token_id: int = None,
            pad_masked_perc: float=0.5,     
        ):

        self.data = np.memmap(memmap_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
        self.is_diffusion_training = is_diffusion_training
        self.is_padded_dataset = is_padded_dataset
        self.stride = context_length if is_padded_dataset else 1
        self.pad_masked_perc = pad_masked_perc
        
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

            # if predicting pad token, make sure to have a balanced mask
            if self.is_padded_dataset:
                
                # Compute what fraction of masked tokens should be pad / nonpad
                t_pad = t * self.pad_masked_perc
                t_nonpad = t * (1-self.pad_masked_perc)
                
                # Get the number
                num_pad = int(t_pad * X.shape[0])
                num_nonpad = int(t_nonpad * X.shape[0])
                
                # Get ids in the X tensor
                pad_ids = np.where(X == self.pad_token_id)[0]
                nonpad_ids = np.where(X != self.pad_token_id)[0]

                # Randomly pick
                np.random.shuffle(pad_ids)
                np.random.shuffle(nonpad_ids)

                # Invert the mask (now all non-masked)
                mask = ~mask
        
                # Select tokens to mask based on pad/nonpad proportions
                if len(pad_ids) > 0 and num_pad > 0:
                    selected_pad_ids = pad_ids[:num_pad]
                    mask[selected_pad_ids] = True
                    
                if len(nonpad_ids) > 0 and num_nonpad > 0:
                    selected_nonpad_ids = nonpad_ids[:num_nonpad]
                    mask[selected_nonpad_ids] = True
            else:
                # Randomly sample the masked ones
                mask = torch.rand(
                    size=(y.shape[0],), 
                    dtype=torch.float32
                ) < t

            # make sure at least one is masked
            if not torch.any(mask):
                idx = torch.randint(0, mask.shape[0], size=(1,)).item()
                mask[idx] = True
                
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
                is_padded_dataset= self.config["padded_dataset"],
                eos_token_id=self.config["eos_token_id"],
                pad_token_id=self.config["pad_token_id"],
                pad_masked_perc=self.config["pad_masked_perc"]
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