import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Tuple
import os

class LengthPredictDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        prompts,
        responses,
        max_length=4096,
        mask_id=126336
    ):
        self.tokenizer = tokenizer
        self.prompt = prompts
        self.response = responses
        self.max_length = max_length
        self.mask_id = mask_id

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, idx):
        prompt = self.prompt[idx]
        response = self.response[idx]

        # 1) render the chat‐template for the prompt alone
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False
        ) # this already includes the assistant role after the user role

        # 2) make the full sequence to measure its length
        full_text = prompt_text + response

        # 3) tokenize just the prompt for inputs
        prompt = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )['input_ids']  # [1, prompt_length]

        x = torch.full((1, self.max_length), self.mask_id, dtype=torch.long) # [1, max_length]
        x[:, :prompt.shape[1]] = prompt.clone()
        

        # 4) re‐tokenize full_text to find where the prompt ends
        prompt_len = len(self.tokenizer(full_text)["input_ids"]) - prompt.shape[1] # this is the length of the prompt
        
        # 5) build eos_positions mask
        eos_positions = torch.tensor(
            [1 if i >= prompt_len else 0 for i in range(self.max_length)],
            dtype=torch.long,
        )

        return {
            "input_ids": x.squeeze(0), # [1, max_length]
            "eos_labels": eos_positions,  # your head can predict this
            "true_length": prompt_len,  # this is the length of the prompt
        }


class DataModule:
    """
    The main datamodule. When iterated over, returns batches of (X, y, msk) of sequence and target sequence shifted by one.
    """
    def __init__(
        self, 
        config,
        tokenizer,
        num_workers=0
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.batch_size = self.config["batch_size"]
        self.context_length = self.config["context_length"]
        self.val_test_perc = self.config["val_test_perc"]
        self.train_dataset: List[Tuple] = None
        self.val_dataset: List[Tuple] = None
        self.test_dataset: List[Tuple] = None

    def setup(self):
        # Split the dataset
        assert self.val_test_perc < 1.0
        # Load the dataset
        path_to_data= os.path.join(    
            os.path.dirname(os.path.dirname(__file__)),
            "data"
        )
        path_to_train = os.path.join(path_to_data, "train.csv")
        path_to_test = os.path.join(path_to_data, "test.csv")
        train_df = pd.read_csv(path_to_train)
        test_df = pd.read_csv(path_to_test)

        # Prepare datasets
        train_prompts = train_df["user_prompt"].tolist()
        train_responses = train_df["model_response"].tolist()
        test_prompts = test_df["user_prompt"].tolist()
        test_responses = test_df["model_response"].tolist()

        # Split train into train/val
        total_train = len(train_df)
        val_size = int(total_train * self.val_test_perc)
        val_size = max(1, val_size)
        train_size = total_train - val_size

        seed = self.config.get("seed", 42)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        self.train_dataset = LengthPredictDataset(
            self.tokenizer,
            [train_prompts[i] for i in train_indices],
            [train_responses[i] for i in train_indices],
            max_length=self.context_length,
        )
        self.val_dataset = LengthPredictDataset(
            self.tokenizer,
            [train_prompts[i] for i in val_indices],
            [train_responses[i] for i in val_indices],
            max_length=self.context_length,
        )
        self.test_dataset = LengthPredictDataset(
            self.tokenizer,
            test_prompts,
            test_responses,
            max_length=self.context_length,
        )
        print(f"Dataset split: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")


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

