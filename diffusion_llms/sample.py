import json
import os
import sys

import tiktoken
import torch
from model import GPT2

# From the command line we can specify the config.file
if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    print("No path/to/config.json provided, defaulting to './config.json'")
    CONFIG_PATH = "./config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Tokenize
tokenizer = tiktoken.get_encoding("gpt2")

# Mask token
mask_token = tokenizer.decode([config["mask_id"]])

# Get prompt
if config["user_prompt"]:
    input_ids = torch.tensor(
        [50256] + tokenizer.encode(config["user_prompt"])
    ).unsqueeze(0)
else:
    input_ids = torch.tensor([50256]).unsqueeze(0)

# Instantiate a model (new or pretrained)
if os.path.exists(config["init_from"]):
    model = GPT2.from_pretrained(config["init_from"])
else:
    model = GPT2(CONFIG_PATH)

# Set evaluation mode
model.eval()

# Generate
n = config["n_samples"]
print(f"\nGenerating {n} samples...\n")
for _ in range(n):
    # List of tensors of shape (B, seq_len)
    xs = model.generate(
        input_ids=input_ids,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        top_k=config["top_k"],
        do_sample=config["do_sample"],
        repetition_penalty=config["repetition_penalty"],
        denoising_strategy=config["denoising_strategy"],
        pipeline=config["pipeline"],
        diffusion_steps=config["diffusion_steps"],
        var_len=config["var_len"],
    )

    # Illustrate the diffusion process
    for x in xs:
        out = tokenizer.decode(x[0].tolist()).replace(mask_token, "<mask>")
        print(out)
        print()
    print("-" * 89)
