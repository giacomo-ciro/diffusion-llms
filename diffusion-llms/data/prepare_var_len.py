"""
Stream the FineWeb dataset from HF, find samples with <context_length 
tokens and save them to a contiguous np.memmap, with appropriate 
padding so they all have context_length total tokens (of which some are
true text tokens and others are eos+pad...).
"""

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
    print("Example: python prepare_var_len.py 1000 ../config.json")
    sys.exit()

# Load configuration
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Get tokenizer
enc = tiktoken.get_encoding("gpt2")
eos_token_id = enc.eot_token  # Usually 50256 for GPT-2
pad_token_id = config.get("pad_token_id", 50257)  # Use a specific pad token if defined
context_length = config.get("context_length", 256)

print(f"Using context_length: {context_length}")
print(f"EOS token ID: {eos_token_id}")
print(f"PAD token ID: {pad_token_id}")

# Create a synthetic dataset from openwebtext
# We'll use existing text as both prompts and answers
dataset = load_dataset(
    "HuggingFaceFW/fineweb",    
    "CC-MAIN-2013-20",  
    split = "train",
    streaming=True,
    trust_remote_code=True,
).shuffle(
    buffer_size=10_000,
    seed = 42,
).take(SAMPLE_SIZE)  

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

# Tokenize the dataset
tokenized = dataset.map(
    process,
    remove_columns=['text'],
)

# Go through the dataset and store all the samples 
# shorter than context_length
print("Counting valid samples...")
tot_len = 0
valid_samples = 0
for sample in tokenized:
    if sample["len"] < context_length:
        tot_len += sample["len"]
        valid_samples += 1
        # print()
        # print(enc.decode(sample["ids"]))

if valid_samples == 0:
    print(f"[!] No samples found with context length = {context_length}.")
    exit()
print(f"Found {valid_samples} samples out of {SAMPLE_SIZE} with < {context_length} tokens")
print(f"Number of text tokens in final dataset: {tot_len:,}")
print(f"Number of total tokens in final dataset: {valid_samples * context_length:,}")
print(f"Average text tokens per sample: {tot_len / valid_samples:,.2f}")

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

# Create memmap object
filename = f'var_len_train_{format_file_size(tot_len)}_{context_length}'
memmap_path = os.path.join(
    os.path.dirname(__file__),
    f"{filename}.bin"
)
memmap_dtype = np.uint16
arr = np.memmap(
    memmap_path,
    dtype=memmap_dtype,
    mode='w+',
    shape=(context_length * valid_samples,)     # blocks of context_length
)

# Go through the dataset again and save the tokens with appropriate padding
idx = 0
for sample in tokenized:

    if sample["len"] < context_length:
        assert sample["ids"][-1] == eos_token_id
        sample_len = sample["len"]
        sample_padded = sample["ids"] + [pad_token_id] * (context_length - sample_len)
        arr[idx: idx+context_length] = np.array(sample_padded, dtype=memmap_dtype)
        idx += context_length
        arr.flush()


# Write metadata file with statistics
with open(os.path.join(os.path.dirname(__file__), f'{filename}.txt'), "w") as f:
    f.write(
f"""Variable Length Dataset (Fixed Structure)
Generated on: {time.strftime("%d-%m-%Y %H:%M:%S")}
Using: $ python improved_prepare_var_len.py {SAMPLE_SIZE} {CONFIG_PATH}
Total number of tokens: {tot_len:,}
Number of samples: {SAMPLE_SIZE}
Context length: {context_length}
Average text length: {tot_len / valid_samples:,.2f}
EOS token id: {eos_token_id}
PAD token id: {pad_token_id}
Format: [text tokens] [{eos_token_id}] [{pad_token_id}, ..., {pad_token_id}]
"""
)

print(f"Variable length dataset saved as {filename}.bin")

# Inspect generate data
print("Inspecting generated memmap...")
n = 5
arr = np.memmap(memmap_path, dtype=memmap_dtype, mode='r')

# Random chunk (multiples of context_len)
idx = np.random.randint(0, valid_samples, size=(n,), dtype=int) * context_length

# Print the text
for i in idx:
    sample = arr[i:i+context_length].tolist()
    # Decoder does not know pad id, replace with eos just for the sake of printing TODO: custom tokenizer
    sample = [token if token != pad_token_id else eos_token_id for token in sample]
    print()
    print(
        enc.decode(sample)
    ) 
print("Done!")