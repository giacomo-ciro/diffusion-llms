import sys
import json
import os

import torch
import tiktoken
from model import GPT2

# From the command line we can specify the config.file
if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    print("No path/to/config.json provided, defaulting to \'./config.json\'")
    CONFIG_PATH = './config.json'

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Tokenize
tokenizer= tiktoken.get_encoding("gpt2")

# Mask token
mask_token = tokenizer.decode([config["mask_id"]])

# Get prompt
if config["user_prompt"]:
    input_ids = torch.tensor(
        [50256] + tokenizer.encode(config["user_prompt"])
    ).unsqueeze(0)
else:
    input_ids = torch.tensor(
        [50256]
    ).unsqueeze(0)

# Load model
model = GPT2(CONFIG_PATH)

# Load model (new or pretrained)
if os.path.exists(config["init_from"]):
    model = GPT2.from_pretrained(config["init_from"])
else:
    model = GPT2(CONFIG_PATH)
    
# Generate
for _ in range(config["n_samples"]):
    
    # List of tensors of shape (B, seq_len)
    xs = model.generate(
        input_ids,
    )

    # Illustrate the diffusion process
    for x in xs:
        out = tokenizer.decode(
            x[0].tolist()
        ).replace(mask_token, "<mask>")
        print(out)
        print()
    print("-"*89)