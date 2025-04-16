# prepare_var_len.py
# Prepares a dataset with prompt+answer+EOS+pad format for variable length generation
import time
import sys
import os
import json
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Command line arguments
if len(sys.argv) >= 3:
    SAMPLE_SIZE = int(sys.argv[1])
    CONFIG_PATH = sys.argv[2]
else:
    print("Usage: python prepare_var_len.py <SAMPLE_SIZE> <CONFIG_PATH>")
    print("Example: python prepare_var_len.py 1000 ../../config.json")
    sys.exit()

# Load configuration
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Get tokenizer
enc = tiktoken.get_encoding("gpt2")
eos_token_id = enc.eot_token  # Usually 50256 for GPT-2
pad_token_id = config.get("pad_token_id", 50257)  # Use a specific pad token if defined

# Create a synthetic dataset from openwebtext
# We'll use existing text as both prompts and answers
print("Loading OpenWebText dataset...")
dataset = load_dataset(
    "openwebtext",
    split="train",
    streaming=True,
).take(SAMPLE_SIZE)

def create_prompt_answer_pairs(example):
    """Split a document into prompt and answer pairs"""
    text = example['text']
    
    # Simple heuristic: split at a sentence boundary near the middle
    split_points = ['. ', '? ', '! ']
    
    # Find all potential split points
    positions = []
    for sep in split_points:
        pos = 0
        while True:
            pos = text.find(sep, pos + 1)
            if pos == -1:
                break
            positions.append(pos + len(sep) - 1)
    
    positions.sort()
    
    # Find a split point near the middle
    if positions:
        mid_point = len(text) // 2
        split_idx = min(positions, key=lambda x: abs(x - mid_point))
        
        prompt = text[:split_idx + 1]
        answer = text[split_idx + 1:].strip()
    else:
        # Fallback: split in the middle of the text
        mid_point = len(text) // 2
        prompt = text[:mid_point]
        answer = text[mid_point:].strip()
    
    return {
        'prompt': prompt,
        'answer': answer
    }

# Process examples into prompt-answer pairs
print("Creating prompt-answer pairs...")
dataset = dataset.map(create_prompt_answer_pairs)

def process_prompt_answer_pair(example):
    """Tokenize prompt and answer with EOS and padding"""
    prompt_tokens = enc.encode_ordinary(example["prompt"])
    answer_tokens = enc.encode_ordinary(example["answer"])
    
    # Add EOS token after the answer
    combined = prompt_tokens + answer_tokens + [eos_token_id]
    
    # Pad to context_length
    if len(combined) < config["context_length"]:
        combined += [pad_token_id] * (config["context_length"] - len(combined))
    else:
        # Truncate if needed
        combined = combined[:config["context_length"]]
    
    # Keep track of important positions
    prompt_len = len(prompt_tokens)
    answer_len = len(answer_tokens)
    eos_position = prompt_len + answer_len
    
    return {
        'ids': combined, 
        'len': len(combined),
        'prompt_len': prompt_len,
        'answer_len': answer_len,
        'eos_position': eos_position
    }

# Tokenize the dataset with prompt+answer+EOS+pad format
print("Tokenizing prompt-answer pairs...")
tokenized = dataset.map(process_prompt_answer_pair)

# Compute total number of tokens
tot_len = 0
for i in tokenized:
    tot_len += i["len"]
print(f"Number of Tokens = {tot_len:,}")

# Create a memmap array
def format_file_size(tot_len):
    if tot_len < 1000:
        return f'{tot_len:.0f}'
    elif tot_len < 1_000_000:
        return f'{tot_len/1e3:.0f}K'
    elif tot_len < 1_000_000_000:
        return f'{tot_len/1e6:.0f}M'
    else:
        return f'{tot_len/1e9:.0f}B'

# Use a distinct filename to indicate variable length dataset
filename = f'var_len_{format_file_size(tot_len)}'
arr = np.memmap(
    os.path.join(os.path.dirname(__file__), f'{filename}.bin'),
    dtype=np.uint16,
    mode='w+',
    shape=(tot_len,)
)

idx = 0
desc = "Writing to variable-length .bin"
for sample in tqdm(tokenized, desc=desc):
    # Write into mmap
    sample_len = sample["len"]
    arr[idx: idx+sample_len] = np.array(sample["ids"])
    idx += sample_len
    arr.flush()

# Write metadata file with statistics
with open(os.path.join(os.path.dirname(__file__), f'{filename}.txt'), "w") as f:
    f.write(
        f"""Variable Length Dataset
Generated on: {time.strftime("%d-%m-%Y %H:%M:%S")}
Using: $ python prepare_var_len.py {SAMPLE_SIZE} {CONFIG_PATH}
Total number of tokens: {tot_len:,}
Format: [prompt tokens] [answer tokens] [EOS token={eos_token_id}] [PAD tokens={pad_token_id}]
"""
    )

print(f"Variable length dataset saved as {filename}.bin")
print(f"Make sure to update your config.json to use this dataset:")
print(f"  \"memmap_path\": \"./data/openwebtext/{filename}.bin\",")
print(f"  \"eos_token_id\": {eos_token_id},")
print(f"  \"pad_token_id\": {pad_token_id},")
print(f"  \"variable_length\": true")