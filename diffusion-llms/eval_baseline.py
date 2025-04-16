
import sys
import torch
import json
import tiktoken
import numpy as np
from model import GPT2

# Load config
if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    CONFIG_PATH = './config.json'

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Tokenizer setup
enc = tiktoken.get_encoding("gpt2")
def encode(x):
    return enc.encode(x, allowed_special={"<|endoftext|>"})

def decode(x):
    # Handle different types of inputs carefully
    if isinstance(x, torch.Tensor):
        if x.dim() > 1:
            # Take first sequence if it's batched
            x = x[0].cpu().numpy().tolist()
        else:
            x = x.cpu().numpy().tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    
    # Make sure x is a flat list of integers
    if isinstance(x, list):
        if isinstance(x[0], list):
            x = x[0]  # Take the first sequence if it's a list of lists
    
    # Ensure all elements are integers
    x = [int(token) for token in x]
    
    return enc.decode(x)

# Test prompts
test_prompts = [
    "Hello, what's your name?",
    "Explain the concept of diffusion models.",
    "Write a short poem about language.",
    "What are the key differences between GPT and BERT?",
    "Calculate 15 * 7 + 22 ="
]

# Load model
model = GPT2(CONFIG_PATH)

# Disable logging temporarily
original_log_method = model.log
model.log = lambda *args, **kwargs: None

print("Baseline DiffuGPT Generation Examples:")
print("-" * 80)

for i, prompt in enumerate(test_prompts):
    print(f"Prompt {i+1}: {prompt}")
    prompt_ids = encode(prompt)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(prompt_ids, max_new_tokens=64)
    
    # Print the shape and type for debugging
    print(f"Output shape: {type(output_ids)}")
    if isinstance(output_ids, torch.Tensor):
        print(f"Tensor shape: {output_ids.shape}")
    elif isinstance(output_ids, list) and output_ids:
        print(f"List length: {len(output_ids)}")
        if isinstance(output_ids[0], list):
            print(f"Inner list length: {len(output_ids[0])}")
    
    # Decode the output
    try:
        output_text = decode(output_ids)
        print(f"Output: {output_text}")
    except Exception as e:
        print(f"Error decoding: {e}")
        print(f"Raw output: {output_ids}")
    
    print("-" * 89)

# Restore original log method
model.log = original_log_method