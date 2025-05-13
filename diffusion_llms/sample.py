import json
import os
import sys

import tiktoken
import torch
from diffusion_llms.models.gpt2_diffusion import DiffuGPT
from diffusion_llms.utils import get_device


# From the command line we can specify the config.file
if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    print("No path/to/config.json provided, defaulting to './config.json'")
    CONFIG_PATH = "./config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

device = get_device()
# Get prompt
if config["user_prompt"]:
    prompt = config["user_prompt"]
else:
    prompt = "What is the capital of France?"



# Tokenize
tokenizer = tiktoken.get_encoding("gpt2")

# Mask token
mask_token = tokenizer.decode([config["mask_id"]])

# Tokenize the prompt
input_ids = torch.tensor(
        [50256] + tokenizer.encode(prompt)
    ).unsqueeze(0).to(device)

# Instantiate a model (new or pretrained)
if os.path.exists(config["init_from"]):
    model = DiffuGPT.from_pretrained(config["init_from"])
else:
    model = DiffuGPT(CONFIG_PATH)

model = model.to(device)
# Set evaluation mode
model.eval()
max_new_tokens = config["max_new_tokens"]
# Generate
n = config["n_samples"]
print(f"\nGenerating {n} samples...\n")
for _ in range(n):

    # List of tensors of shape (B, seq_len)
    xs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=config["temperature"],
        top_k=config["top_k"],
        denoising_strategy=config["denoising_strategy"],
        diffusion_steps=config["diffusion_steps"],
    )

    # Illustrate the diffusion process
    for x in xs:
        out = tokenizer.decode(x[0].tolist()).replace(mask_token, "<mask>")
        print(out)
        print()
    print("-" * 89)
    print(f"generated {max_new_tokens} tokens")
    print("-" * 89)
