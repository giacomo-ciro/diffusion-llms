import sys
import ast

import tiktoken
from model import GPT2

CONFIG_PATH = "./config.json"

if len(sys.argv) == 3:
    prompt = str(sys.argv[1])
    n_sample = ast.literal_eval(sys.argv[2])
    assert (
        isinstance(n_sample, int)
    )
else:
    print(f"[!] Provide prompt and number of responses to generate.")
    sys.exit()

# Tokenize
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
prompt_idx = encode(prompt)

# Load model
model = GPT2(CONFIG_PATH)

# Generate
for _ in range(n_sample):
    ans = model.generate(prompt_idx, max_new_tokens=128)
    ans = decode(ans[0].tolist())
    print(ans)
    print("-"*89)