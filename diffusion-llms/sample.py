import sys
import ast

import tiktoken
from model import GPT2

CONFIG_PATH = "./local_config.json"

# From the command line we can specify the config.file
if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    print("No path/to/config.json provided, defaulting to \'./config.json\'")
    CONFIG_PATH = './config.json'

# Tokenize
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
prompt_idx = encode(config["user_prompt"])

# Load model
model = GPT2(CONFIG_PATH)

# Generate
for _ in range(config["n_samples"]):
    ans = model.generate(prompt_idx, max_new_tokens=128)
    ans = decode(ans[0].tolist())
    print(ans)
    print("-"*89)