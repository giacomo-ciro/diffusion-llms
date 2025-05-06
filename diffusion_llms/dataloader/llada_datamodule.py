import json
import os

import torch
import numpy as np
import lightning as pl
from torch.utils.data import Dataset, DataLoader
import sys

import tiktoken


class RegressionDataset(Dataset):
    def __init__(self, config: dict, tokenizer: tiktoken.Encoding):
        super().__init__()
        self.tokenizer = tokenizer
        self.csv_path_rel = config["llada_train_path"]
        
        self.prompts: list[str] = self.data_df["user_prompt"].astype(str).tolist()
        self.answers: list[str] = self.data_df["model_response"].astype(str).tolist()

        self.length = len(self.prompts)
        print(f"RegressionDataset: Initialized with {self.length} samples.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves and processes a prompt-response pair at the specified index.
        
        This method prepares data for the regression task where the model predicts
        the length of the answer. It tokenizes both prompt and answer, creates an
        input sequence (X) by combining them with appropriate padding/truncation,
        creates a target value (y) representing the answer length, and generates
        a mask (msk) identifying answer tokens in the input sequence.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: Contains:
                - X (torch.Tensor): Input token sequence [context_length]
                - y (torch.Tensor): Target answer length [1]
                - msk (torch.Tensor): Boolean mask marking answer tokens [context_length]
        """
        prompt_str: str = self.prompts[idx]
        answer_str: str = self.answers[idx]
        eos_token_id: int = self.config.get("eos_token_id", 50256)
        pad_token_id: int = self.config.get("pad_token_id", 50257)
        context_length: int = self.config.get("context_length", 1024)

        # Tokenize the prompt and answer strings
        tokenized_prompt: list[int] = self.tokenizer.encode(prompt_str)
        tokenized_answer: list[int] = self.tokenizer.encode(answer_str)

        # Prepare the target 'y' for regression: the length of the answer including an EOS token
        tokenized_answer_with_eos: list[int] = tokenized_answer + [eos_token_id]
        y_value: int = len(tokenized_answer_with_eos)
        y: torch.Tensor = torch.tensor([y_value], dtype=torch.float) 

        # Prepare the input sequence 'X' (prompt + answer_with_eos, padded/truncated)
        X_token_ids: list[int] = [pad_token_id] * context_length
        
        len_prompt: int = len(tokenized_prompt)
        # len_answer_eos is y_value (already calculated)

        actual_len_prompt_in_X: int = 0
        actual_len_answer_in_X: int = 0

        # Handle the case where the prompt is longer than the context length
        # If the prompt is longer than the context length, truncate it
        # If the prompt is shorter than the context length, fill the rest with the answer
        if len_prompt >= context_length:
            X_token_ids[:context_length] = tokenized_prompt[:context_length]
            actual_len_prompt_in_X = context_length
            # actual_len_answer_in_X remains 0
        else:
            X_token_ids[:len_prompt] = tokenized_prompt
            actual_len_prompt_in_X = len_prompt
            
            remaining_space: int = context_length - len_prompt
            num_answer_tokens_to_copy: int = min(remaining_space, y_value) 
            
            if num_answer_tokens_to_copy > 0:
                X_token_ids[len_prompt : len_prompt + num_answer_tokens_to_copy] = \
                    tokenized_answer_with_eos[:num_answer_tokens_to_copy]
                actual_len_answer_in_X = num_answer_tokens_to_copy

        X: torch.Tensor = torch.tensor(X_token_ids, dtype=torch.long)

        # Prepare the mask 'msk' indicating the answer part in 'X'
        msk_np = np.zeros(context_length, dtype=bool)
        ans_start_idx_in_X: int = actual_len_prompt_in_X
        ans_end_idx_in_X: int = actual_len_prompt_in_X + actual_len_answer_in_X
        
        if actual_len_answer_in_X > 0:
            msk_np[ans_start_idx_in_X:ans_end_idx_in_X] = True
        
        msk: torch.Tensor = torch.from_numpy(msk_np)

        # ===== TEMPLATE =====
        # # Get elements at idx
        # prompt = self.prompts[idx]
        # ans = self.answers[idx]

        # # Concat & create mask
        # X = concat(prompt, ans)
        # y = len(ans)
        # msk = torch.zeros_like(X)
        # # add ones at answer + eos positions
        
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