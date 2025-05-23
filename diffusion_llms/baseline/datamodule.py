import torch
import torch.nn as nn
from torch.utils.data import Dataset



class PromptDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        enc = self.tokenizer(prompt, padding='max_length', truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
def get_length(responses, tokenizer, max_length, steps):
    lengths = []
    for response in responses:
        enc = tokenizer(response, padding=False, truncation=True, max_length=max_length)
        length = len(enc['input_ids'])
        for i, step in enumerate(steps):
            if length <= step:
                lengths.append(i)
                break
    return lengths


def get_length_reg(responses, tokenizer):
    lengths = []
    for response in responses:
        enc = tokenizer(response, padding=False)
        lengths.append(len(enc['input_ids']))
    return lengths