import numpy as np
import tiktoken

# Load the memmap file
data = np.memmap('./var_len_26K.bin', dtype=np.uint16, mode='r')

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eos_token_id = enc.eot_token  # 50256
pad_token_id = 50257  # From your config

print(f"Total tokens in dataset: {len(data)}")
print(f"EOS token ID: {eos_token_id}")
print(f"PAD token ID: {pad_token_id}")

# Count occurrences of special tokens
eos_count = np.sum(data == eos_token_id)
pad_count = np.sum(data == pad_token_id)
print(f"EOS token count: {eos_count}")
print(f"PAD token count: {pad_count}")

# Check sample sequences
sequence_length = 256  # Same as your context_length
num_samples = 100

for i in range(num_samples):
    # Get a sequence
    start_idx = i * sequence_length
    end_idx = start_idx + sequence_length
    seq = data[start_idx:end_idx]
    
    # Find special tokens in this sequence
    eos_positions = np.where(seq == eos_token_id)[0]
    pad_positions = np.where(seq == pad_token_id)[0]
    
    if len(eos_positions) == 0 and len(pad_positions) == 0:
        continue
    
    print(f"\n--- Sample {i+1} ---")
    if len(eos_positions) > 0:
        print(f"EOS positions: {eos_positions}")
    if len(pad_positions) > 0:
        print(f"PAD positions: {pad_positions}")
    
    # Decode and print the sequence (excluding pad tokens)
    # Replace pad tokens with a value the tokenizer can handle (e.g., 0)
    valid_seq = [t if t != pad_token_id else 0 for t in seq]
    
    # For a cleaner view, only decode up to the first EOS or PAD token
    cutoff = min(eos_positions[0] if len(eos_positions) > 0 else sequence_length,
                pad_positions[0] if len(pad_positions) > 0 else sequence_length)
    
    text = enc.decode(valid_seq[:cutoff])
    print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
    
    # Print structure analysis
    if len(eos_positions) > 0 and len(pad_positions) > 0:
        # Check if structure follows [content][EOS][PAD]... pattern
        if eos_positions[0] < pad_positions[0] and pad_positions[0] == eos_positions[0] + 1:
            print("✓ Valid structure: content→EOS→PAD")
        else:
            print("✗ Unexpected structure")
    elif len(eos_positions) > 0:
        print("! Has EOS but no PAD tokens")
    elif len(pad_positions) > 0:
        print("! Has PAD but no EOS tokens")