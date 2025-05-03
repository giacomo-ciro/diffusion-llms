"""
Stream the FineWeb dataset from HF, find samples with context_length 
tokens and save them to two contiguous np.memmap, one for train and 
the other for test, with appropriate padding so they all have context_length 
total tokens (of which some are true text tokens and others are eos+pad...).
"""

import time
import os
import json

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import argparse

# Name of this file
this_script_name = os.path.basename(__file__)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="path/to/config.json")
parser.add_argument("-tr", "--train", type=int, default=10, help="Number of documents from FineWeb to build train dataset.")
parser.add_argument("-te", "--test", type=int, default=1, help="Number of documents from FineWeb to build test dataset.")
parser.add_argument("-n", type=int, default=3, help="How many samples to inspect at the end.")
args = parser.parse_args()

# Load configuration
with open(args.config, "r") as f:
    config = json.load(f)

# Unique ID for this pre-processing job
puid = time.strftime("%d%m%Y%H%M%S")
os.mkdir(f"{puid}")
puid_folder = os.path.join(
    os.path.dirname(__file__),
    f"{puid}"
)

# Get tokenizer
enc = tiktoken.get_encoding("gpt2")
# however, we will define some special tokens to tackle the variable length
eos_token_id = config["eos_token_id"]  # Usually 50256 for GPT-2
pad_token_id = config["pad_token_id"]
context_length = config.get("context_length", 256)

print(f"context_length: {context_length}")
print(f"EOS token ID: {eos_token_id}")
print(f"PAD token ID: {pad_token_id}")
print(f"Train documents: {args.train}")
print(f"Test documents: {args.test}")

# Create a synthetic dataset from openwebtext
# We'll use existing text as both prompts and answers
dataset = load_dataset(
    "HuggingFaceFW/fineweb",    
    # "CC-MAIN-2013-20",  
    split = "train",        # unfortunately, only train available...otherwise it would have been easier to prepare trian/test
    streaming=True,
    trust_remote_code=True,
).shuffle(
    buffer_size=10_000,
    seed = 42,
)

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
tot_len_train = 0
tot_len_test = 0
valid_samples_train = 0
valid_samples_test = 0
checked_samples = 0
pbar = tqdm(tokenized, desc = "Counting valid samples", unit="samples")
for sample in pbar:

    # Keep track of checked samples
    checked_samples += 1
    
    # When we find something useful
    if sample["len"] < context_length:

        # First fill in train
        if valid_samples_train < args.train:
            tot_len_train += sample["len"]
            valid_samples_train += 1
        # Then fill in test
        else:
            tot_len_test += sample["len"]
            valid_samples_test += 1

    # Update progress bar
    pbar.set_postfix(
        valid_samples_train = f"{valid_samples_train:,}",
        valid_samples_test = f"{valid_samples_test:,}"
    )

    # Stop when we have enough samples
    if valid_samples_test >= args.test:
        break

# Print some information about the first sweep
if valid_samples_train == 0:
    print(f"[!] No samples found with context length = {context_length}.")
    exit()

# Stats of just performed check
print(f"""
Checked: {checked_samples}

== Train ==
Valid (target): {valid_samples_train} ({args.train})
Text Tokens: {tot_len_train:,}
Tot Tokens: {valid_samples_train * context_length:,}
Average Text Tokens per Sample: {tot_len_train / valid_samples_train:,.2f}

== Test ==
Valid (target): {valid_samples_test} ({args.test})
Text Tokens: {tot_len_test:,}
Tot Tokens: {valid_samples_test * context_length:,}
Average Text Tokens per Sample: {tot_len_test / valid_samples_test:,.2f}

"""
)

# Format the name of the memmap file
def format_file_size(tot_len):
    if tot_len < 1000:
        return f'{tot_len:.0f}'
    elif tot_len < 1_000_000:
        return f'{tot_len/1e3:.0f}K'
    elif tot_len < 1_000_000_000:
        return f'{tot_len/1e6:.0f}M'
    else:
        return f'{tot_len/1e9:.0f}B'

# Dtype
memmap_dtype = np.uint16

# train
filename_train = f'var_len_train_{format_file_size(tot_len_train)}_{context_length}'
memmap_path_train = os.path.join(
    puid_folder,
    f"{filename_train}.bin"
)
arr_train = np.memmap(
    memmap_path_train,
    dtype=memmap_dtype,
    mode='w+',
    shape=(context_length * valid_samples_train,)     # blocks of context_length
)

# Test
filename_test = f'var_len_test_{format_file_size(tot_len_test)}_{context_length}'
memmap_path_test = os.path.join(
    puid_folder,
    f"{filename_test}.bin"
)
arr_test = np.memmap(
    memmap_path_test,
    dtype=memmap_dtype,
    mode='w+',
    shape=(context_length * valid_samples_test,)     # blocks of context_length
)

# Go through the dataset again and save the tokens with appropriate padding
idx = 0
valid_samples_train = 0
valid_samples_test = 0
pbar = tqdm(tokenized, desc = "Saving valid samples", unit="samples")
for sample in pbar:
        # Keep track of checked samples
    checked_samples += 1
    
    # When we find something useful
    if sample["len"] < context_length:

        # Check correct build
        assert sample["ids"][-1] == eos_token_id

        # Get data
        sample_len = sample["len"]
        sample_padded = sample["ids"] + [pad_token_id] * (context_length - sample_len)

        # First fill in train
        if valid_samples_train < args.train:
            arr = arr_train
            tot_len_train += sample_len
            valid_samples_train += 1
        # Then fill in test
        else:
            arr = arr_test
            idx = 0 if valid_samples_test == 0 else idx # reset index
            tot_len_test += sample_len
            valid_samples_test += 1

        # Copy to correct array
        assert idx < len(arr)
        arr[idx: idx+context_length] = np.array(sample_padded, dtype=memmap_dtype)
        idx += context_length
        arr.flush()

    # Update progress bar
    pbar.set_postfix(
        valid_samples_train = f"{valid_samples_train:,}",
        valid_samples_test = f"{valid_samples_test:,}"
    )

    # Stop when we have enough samples
    if valid_samples_test >= args.test:
        break

# Write metadata file with statistics
metadata_path = os.path.join(
    puid_folder,
    f'{puid}.txt'
)
with open(metadata_path, "w") as f:
    f.write(
f"""Metadata for Test / Train Datasets {puid}

Generated on: {puid}
Using: $ python {this_script_name} --config {args.config} --train {args.train} --test {args.test}

Total Checked Samples: {checked_samples}

== Train ==
Valid (target): {valid_samples_train} ({args.train})
Text Tokens: {tot_len_train:,}
Tot Tokens: {valid_samples_train * context_length:,}
Average Text Tokens per Sample: {tot_len_train / valid_samples_train:,.2f}

== Test ==
Valid (target): {valid_samples_test} ({args.test})
Text Tokens: {tot_len_train:,}
Tot Tokens: {valid_samples_test * context_length:,}
Average Text Tokens per Sample: {tot_len_test / valid_samples_test:,.2f}

== Hyper-params ==
EOS token id: {eos_token_id}
PAD token id: {pad_token_id}
Format: [text tokens] [{eos_token_id}] [{pad_token_id}, ..., {pad_token_id}]

"""
    )

print()
print(f"Train saved to {memmap_path_train}")
print(f"Test saved to {memmap_path_train}")
print(f"Metadata saved to {metadata_path}")

# Inspect generated data
print("Inspecting generated train memmap...")
arr = np.memmap(memmap_path_train, dtype=memmap_dtype, mode='r')

# Random chunk (multiples of context_len)
idx = np.random.randint(0, valid_samples_train, size=(args.n,), dtype=int) * context_length

# Print the text
for i in idx:
    sample = arr[i:i+context_length].tolist()
    # Decoder does not know pad id, replace with eos just for the sake of printing TODO: custom tokenizer
    sample = [token if token != pad_token_id else eos_token_id for token in sample]
    print()
    print(
        enc.decode(sample)
    ) 

print(
"""
The job is done!
Don\'t worry if you get \"Fata Python Error\" after this message...I don't know why it happens
        
        ~ Jack :)
        
"""
)