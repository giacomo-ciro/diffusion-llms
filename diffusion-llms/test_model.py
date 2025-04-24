import os
import time
import sys
import json
import wandb
import lightning as pl
from model import GPT2
from datamodule import MemmapDataModule
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils import check_config_validity
import tiktoken
import torch

def main():
    # From the command line we can specify the config.file
    if len(sys.argv) == 2:
        CONFIG_PATH = sys.argv[1]
    else:
        print("No path/to/config.json provided, defaulting to \'./config.json\'")
        CONFIG_PATH = './config.json'

    # Configuration file
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Check validity of configuration
    check_config_validity(config)


    # Tokenize
    tokenizer= tiktoken.get_encoding("gpt2")

    # Mask token
    mask_token = tokenizer.decode([config["mask_id"]])

    # Get prompt
    input_ids = torch.tensor(
        [50256] + tokenizer.encode(config["user_prompt"])
    ).unsqueeze(0)

    # model
    model = GPT2(CONFIG_PATH)
    model = torch.compile(model).to('cuda')

    
    input_ids = input_ids.to('cuda')
    max_new_tokens = 100
    temperature = 1.0
    top_k = None
    # Generate text using the model
    res = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    for x in res:
        out = tokenizer.decode(
            x[0].tolist()
        ).replace(mask_token, "<mask>")
        print(out)
    print("-"*89)



if __name__ == "__main__":
    main()