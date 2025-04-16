# inspect_dataset.py
import numpy as np
import tiktoken

# Load the memmap file
data = np.memmap('./var_len_26K.bin', dtype=np.uint16, mode='r')

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eos_token_id = enc.eot_token

# Count occurrences of EOS token
eos_count = np.sum(data == eos_token_id)
print(f"EOS token count: {eos_count}")

# Check sample sequences
sequence_length = 256  # Same as your context_length
num_samples = 100

for i in range(num_samples):
    # we first check if eos_token is present in the sample
    start_idx = i * sequence_length
    end_idx = start_idx + sequence_length
    seq = data[start_idx:end_idx]
    #print(f"Sample {i+1}: {seq}")
    
    # Find EOS tokens in this sequence
    eos_positions = np.where(seq == eos_token_id)[0]

    if len(eos_positions) == 0:
        continue
    
    print(f"\nSample {i+1}:")
    print(f"EOS positions: {eos_positions}")
    
    # Decode and print the sequence
    text = enc.decode(seq.tolist())
    print(f"Text: {text[:100]}...")  # First 100 chars