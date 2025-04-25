import time
import sys
import os
import json

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# Command line arguments
if len(sys.argv) == 3:
    SAMPLE_SIZE = int(sys.argv[1])
    CONFIG_PATH = sys.argv[2]
else:
    print("Usage: python prepare_var_len.py <SAMPLE_SIZE> <CONFIG_PATH>")
    print("Example: python prepare_var_len.py 1000 ../config.json")
    sys.exit()

# Load configuration
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
# however, we will define some special tokens to tackle the variable length
eos_token_id = enc.eot_token  # Usually 50256 for GPT-2
pad_token_id = 50257
context_length = config.get("context_length", 256)

print(f"Using context_length: {context_length}")
print(f"EOS token ID: {eos_token_id}")
print(f"PAD token ID: {pad_token_id}")

# Create a synthetic dataset from openwebtext
# We'll use existing text as both prompts and answers
# this won't run if a subdir called openwebtext already exists in the executing directory
print("Loading OpenWebText dataset...")
dataset = load_dataset(
    "openwebtext",
    split="train",
    streaming=True,
).take(SAMPLE_SIZE)

def create_prompt_answer_pairs(example):
    """Split a document into prompt and answer pairs"""
    text = example['text'] # we don't use encode_ordinary here, we want the raw text
    # Simple heuristic: split at a sentence boundary near the middle
    split_points = ['. ', '? ', '! ']
    positions = []
    for sep in split_points:
        pos = 0
        while True:
            pos = text.find(sep, pos + 1) 
            if pos == -1:
                break
            positions.append(pos + len(sep) - 1)
    positions.sort()
    print(positions)
    
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
    
    # Calculate maximum content length to ensure padding
    # Reserve at least 20% of context for padding
    max_content_length = int(context_length * 0.8)
    
    # If combined content is too long, truncate but preserve both prompt and answer
    combined_content_length = len(prompt_tokens) + len(answer_tokens)
    if combined_content_length > max_content_length - 1:  # -1 for EOS token
        # Keep at least 25% of the prompt and 25% of the answer
        min_prompt_length = max(1, len(prompt_tokens) // 4)
        min_answer_length = max(1, len(answer_tokens) // 4)
        
        # Calculate remaining space after ensuring minimums
        remaining = max_content_length - 1 - min_prompt_length - min_answer_length
        
        # Allocate remaining space proportionally
        if remaining > 0:
            prompt_ratio = len(prompt_tokens) / combined_content_length
            additional_prompt = int(remaining * prompt_ratio)
            
            prompt_length = min_prompt_length + additional_prompt
            answer_length = max_content_length - 1 - prompt_length
        else:
            prompt_length = min_prompt_length
            answer_length = min_answer_length
    
        # Randomly truncate by 0-10% more to add variety to EOS positions
        if combined_content_length > 100:  # Only for longer contents
            random_trunc = np.random.randint(0, int(combined_content_length * 0.1))
            if len(answer_tokens) > random_trunc + 10:  # Ensure we don't remove too much from answer
                answer_tokens = answer_tokens[:-random_trunc]
            
        prompt_tokens = prompt_tokens[:prompt_length]
        answer_tokens = answer_tokens[:answer_length]
    
    # Create the final sequence with EOS and padding
    combined = prompt_tokens + answer_tokens + [eos_token_id]
    
    # Add padding to reach context_length
    padding_needed = context_length - len(combined)
    combined += [pad_token_id] * padding_needed
    
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

# Verify dataset structure
print("Verifying dataset structure...")
eos_count = 0
pad_count = 0
correct_structure = 0

for sample in tokenized:
    ids = sample['ids']
    eos_positions = [i for i, x in enumerate(ids) if x == eos_token_id]
    pad_positions = [i for i, x in enumerate(ids) if x == pad_token_id]
    
    if eos_positions:
        eos_count += 1
        if pad_positions and pad_positions[0] == eos_positions[-1] + 1:
            correct_structure += 1
            pad_count += 1

print(f"Number of samples with EOS token: {eos_count} out of {SAMPLE_SIZE}")
print(f"Number of samples with PAD tokens: {pad_count} out of {SAMPLE_SIZE}")
print(f"Number of samples with correct EOS->PAD structure: {correct_structure} out of {SAMPLE_SIZE}")

if eos_count < SAMPLE_SIZE * 0.9 or correct_structure < SAMPLE_SIZE * 0.9:
    print("WARNING: Many samples are missing EOS tokens or correct structure. Check the dataset creation process.")
    if input("Continue anyway? (y/n): ").lower() != 'y':
        sys.exit()

# Prepare dataset in a format suitable for fixed-length retrieval
# This is critical: we flatten the dataset but ensure each sample is context_length long
# This allows the dataloader to efficiently load fixed-size chunks
full_dataset = []
for sample in tokenized:
    full_dataset.extend(sample['ids'])

tot_len = len(full_dataset)
print(f"Number of Tokens = {tot_len:,}")
print(f"Expected number based on samples * context_length = {SAMPLE_SIZE * context_length}")

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

# Use a distinct filename to indicate variable length dataset with improvements
filename = f'var_len_fixed_{format_file_size(tot_len)}'
arr = np.memmap(
    os.path.join(os.path.dirname(__file__), f'openwebtext_local/{filename}.bin'),
    dtype=np.uint16,
    mode='w+',
    shape=(tot_len,)
)


# Write the entire dataset at once
arr[:] = np.array(full_dataset, dtype=np.uint16)
arr.flush()

# Write metadata file with statistics
with open(os.path.join(os.path.dirname(__file__), f'openwebtext_local/{filename}.txt'), "w") as f:
    f.write(
f"""Variable Length Dataset (Fixed Structure)
Generated on: {time.strftime("%d-%m-%Y %H:%M:%S")}
Using: $ python improved_prepare_var_len.py {SAMPLE_SIZE} {CONFIG_PATH}
Total number of tokens: {tot_len:,}
Number of samples: {SAMPLE_SIZE}
Context length: {context_length}
Number of samples with EOS token: {eos_count}
Number of samples with PAD tokens: {pad_count}
Number of samples with correct EOS->PAD structure: {correct_structure}
Format: [prompt tokens] [answer tokens] [EOS token={eos_token_id}] [PAD tokens={pad_token_id}]
"""
    )

print(f"Variable length dataset saved as {filename}.bin")
print(f"Make sure to update your config.json to use this dataset:")
print(f"  \"memmap_path\": \"./data/openwebtext_local/{filename}.bin\",")
print(f"  \"eos_token_id\": {eos_token_id},")
print(f"  \"pad_token_id\": {pad_token_id},")
print(f"  \"variable_length\": true")