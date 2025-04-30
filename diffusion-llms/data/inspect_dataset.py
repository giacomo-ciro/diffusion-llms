import numpy as np
import tiktoken

# Load the memmap file
memmap_path = "var_len_train_16K_500.bin"
memmap_dtype = np.uint16
context_length = 500
arr = np.memmap(memmap_path, dtype=memmap_dtype, mode='r')

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eos_token_id = enc.eot_token  # 50256
pad_token_id = 50257  # From your config
tot_samples = len(arr) / context_length
print(f"Total tokens in dataset: {len(arr):,}")
print(f"Total samples: {tot_samples:,}")
print(f"Context Length: {context_length}")
print(f"EOS token ID: {eos_token_id}")
print(f"PAD token ID: {pad_token_id}")
print(f"Non-PAD/EOS tokens: {np.sum((arr != eos_token_id) & (arr != pad_token_id)):,}")
print()

# Inspect generate data
print("Sampling from generate memmap...")

while input("Press enter to sample from dataset (any other key to exit): ") == "":
    # Random chunk (multiples of context_len)
    idx = np.random.randint(0, tot_samples, size=(1,), dtype=int).item() * context_length

    sample = arr[idx:idx+context_length].tolist()
    # Decoder does not know pad id, replace with eos just for the sake of printing TODO: custom tokenizer
    sample = [token if token != pad_token_id else eos_token_id for token in sample]
    decoded = enc.decode(sample).replace("<|endoftext|>", "eos ")
    print("\nTokens:\n", sample)
    print()
    print("Decoded:\n",
        decoded
    ) 
    print()
print("Done!")