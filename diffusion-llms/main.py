import json
import wandb
import torch
from model import GPT2
import tiktoken

CONFIG_PATH = "./config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

model = GPT2(CONFIG_PATH)

# Test model Logic
# 1 sentence of len 10, from a vocab of 100 toks
T = 10
sentence = torch.randint(0, 100, size=(1, T+1))
X = sentence[:, :-1]
y = sentence[:, 1:]

out = model.step(
    batch = (X, y),     # read in as idx, targets = batch
    batch_idx = 0  
)

# The loss
print(f"Loss {out}")

# TODO: 
# 1. Implement tokenizer (inside lightning module)
# 2. Set up lightning training