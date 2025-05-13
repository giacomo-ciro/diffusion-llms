import pandas as pd
import torch
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import pytorch_lightning as pl


class EmbeddedDataset(Dataset):
    """
    Dataset that loads precomputed embeddings from H5 files.
    """
    def __init__(self, h5_file_path, context_length=4096):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.context_length = context_length
        
        # Get metadata about the dataset
        with h5py.File(self.h5_file_path, 'r') as f:
            self.num_batches = len(f['embeddings'])
            
            # Calculate total examples
            self.total_items = 0
            self.batch_item_counts = []
            
            for i in range(self.num_batches):
                batch_size = f[f'embeddings/batch_{i}/last_hidden'].shape[0]
                self.batch_item_counts.append(batch_size)
                self.total_items += batch_size
                
            # Get dimensions for initialization
            self.hidden_dim = f['embeddings/batch_0/last_hidden'].shape[2]
    
    def __len__(self):
        return self.total_items
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.
        Maps the flat index to the correct batch and item within that batch.
        """
        # Find which batch contains this index
        batch_idx = 0
        local_idx = idx
        
        with h5py.File(self.h5_file_path, 'r') as f:
            # Find the correct batch and local index
            for i, count in enumerate(self.batch_item_counts):
                if local_idx < count:
                    batch_idx = i
                    break
                local_idx -= count
            
            # Load data from the correct batch
            last_hidden = torch.tensor(f[f'embeddings/batch_{batch_idx}/last_hidden'][local_idx])
            pooled = torch.tensor(f[f'embeddings/batch_{batch_idx}/pooled'][local_idx])
            eos_labels = torch.tensor(f[f'labels/batch_{batch_idx}/eos_labels'][local_idx])
            true_length = torch.tensor(f[f'labels/batch_{batch_idx}/true_lengths'][local_idx])
            
            # Create attention mask (all ones since we're using precomputed embeddings)
            attention_mask = torch.ones(last_hidden.shape[0], dtype=torch.long)
        
        return {
            "last_hidden": last_hidden,          # [seq_len, hidden_dim]
            "pooled": pooled,                    # [hidden_dim]
            "eos_labels": eos_labels,            # [seq_len]
            "true_length": true_length,          # [1]
            "attention_mask": attention_mask     # [seq_len]
        }


class EmbeddedDataModule:
    """
    Data module that loads precomputed embeddings from H5 files.
    """
    def __init__(
        self, 
        config,
        embedded_data_dir,
        num_workers=4
    ):
        self.config = config
        self.embedded_data_dir = embedded_data_dir
        self.num_workers = num_workers
        self.batch_size = self.config.get("batch_size", 32)
        self.context_length = self.config.get("context_length", 4096)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage=None):
        """
        Load datasets from H5 files.
        """
        # Define file paths
        train_file = os.path.join(self.embedded_data_dir, "train_embeddings.h5")
        val_file = os.path.join(self.embedded_data_dir, "val_embeddings.h5")
        test_file = os.path.join(self.embedded_data_dir, "test_embeddings.h5")
        
        # Check if files exist
        if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
            missing = [f for f in [train_file, val_file, test_file] if not os.path.exists(f)]
            raise FileNotFoundError(f"Missing embedding files: {missing}")
        
        # Create datasets
        self.train_dataset = EmbeddedDataset(train_file, self.context_length)
        self.val_dataset = EmbeddedDataset(val_file, self.context_length)
        self.test_dataset = EmbeddedDataset(test_file, self.context_length)
        
        print(f"Dataset loaded: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
    
    def collate_fn(self, batch):
        """
        Custom collate function for batching items of variable length.
        """
        last_hidden = torch.stack([item["last_hidden"] for item in batch])
        pooled = torch.stack([item["pooled"] for item in batch])
        eos_labels = torch.stack([item["eos_labels"] for item in batch])
        true_length = torch.stack([item["true_length"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        return {
            "last_hidden": last_hidden,     # [batch_size, seq_len, hidden_dim]
            "pooled": pooled,               # [batch_size, hidden_dim]
            "eos_labels": eos_labels,       # [batch_size, seq_len]
            "true_length": true_length,     # [batch_size, 1]
            "attention_mask": attention_mask # [batch_size, seq_len]
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn
        )