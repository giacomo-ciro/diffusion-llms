# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
import time
import sys
import os
import json
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# From the command line we can specify the config.file
if len(sys.argv) == 2:
    SAMPLE_SIZE = int(sys.argv[1])
else:
    print("Please, provide sample size. For example, run \"python prepare.py 100\"")
    sys.exit()

num_proc = os.cpu_count()

# full datast takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset(
    "openwebtext",
    split = "train",
    streaming=True,
).take(SAMPLE_SIZE)


# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# Tokenize the dataset
tokenized = dataset.map(
    process,
    remove_columns=['text'],
)

# Compute total number of tokens
tot_len = 0
for i in tokenized:
    tot_len += i["len"]
    # print(i)
print(f"Number of Tokens = {tot_len:,}")

# Create a memmap array
# File name
def format_file_size(tot_len):
    if tot_len < 1000:
        return f'{tot_len:.0f}'
    elif tot_len < 1_000_000:
        return f'{tot_len/1e3:.0f}K'
    elif tot_len < 1_000_000_000:
        return f'{tot_len/1e6:.0f}M'
    else:
        return f'{tot_len/1e9:.0f}B'
filename = f'train_{format_file_size(tot_len)}'
arr = np.memmap(
    os.path.join(os.path.dirname(__file__), f'{filename}.bin'),
    dtype=np.uint16,
    mode='w+',
    shape=(tot_len,)
)

idx = 0
desc = "Writing to .bin (ETA not available because we are working with IterableDataset)"
for sample in tqdm(tokenized, desc=desc):

    # Write into mmap
    sample_len = sample["len"]
    arr[idx: idx+sample_len] = np.array(sample["ids"])
    idx += sample_len
    arr.flush()

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')

with open(os.path.join(os.path.dirname(__file__), f'{filename}.txt'), "w") as f:
    f.write(
        f"""Generated on: {time.strftime("%d-%m-%Y %H:%M:%S")}
        Using: $ python prepare.py {SAMPLE_SIZE}
        Tot number of tokens: {tot_len:,}
        """
    )