import numpy as np
from diffusion_llms.tokenizers.custom_gpt_w_pad import CustomGPT2TokenizerWithPad

# Load the memmap file
memmap_path = "/Users/vittorio/Projects/uni/Deep learning/diffusion-llms/diffusion_llms/data/05052025151606_old/train.bin"
memmap_dtype = np.uint16
context_length = 1024
arr = np.memmap(memmap_path, dtype=memmap_dtype, mode='r')

# Initialize custom tokenizer
enc = CustomGPT2TokenizerWithPad()
eos_token_id = enc.tokenizer.eot_token  # 50256
pad_token_id = enc.pad_token_id  # From custom tokenizer

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
    # Decoder does not know pad id, replace with eos just for the sake of printing
    
    decoded = enc.decode(sample)
    print("\nTokens:\n", sample)
    print()
    print("Decoded:\n",
        decoded
    ) 
    print()
print("Done!")