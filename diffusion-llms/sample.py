import sys
import json

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
enc = tiktoken.get_encoding("gpt2")
def encode(x):
    return enc.encode(x, allowed_special={"<|endoftext|>"})
def decode(x):
    return enc.decode(x)
prompt_idx = encode(config["user_prompt"])

# Load model
model = GPT2(CONFIG_PATH)

# Generate
for _ in range(config["n_samples"]):
    ans = model.generate(
        prompt_idx, 
        max_new_tokens=config.get("max_new_tokens", 128),
        temperature=config.get("temperature", 1.0),
        top_k=config.get("top_k", None)
    )
    ans = decode(ans[0].tolist())
    print(ans)
    print("-"*89) # mirko alessandrini reference?