"""
Make plots on the eos distribution. X-axis token position, Y-axis probability of eos.
Plot the average for 10 random plots with / without trigger words.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
import re
import numpy as np
import sys
sys.path.append('.')

from diffusion_llms.models.llada import LladaBackbone
import torch
torch.mps.empty_cache()

device = 'mps'  # or 'cuda' for GPU, 'cpu' for CPU

# Plot logits of eos

TRIGGER_WORDS = ["explain", "tell me"]
N = 10
seq_len = 1024
mask_id = 126336

print("Loading data...")
# Train prompts
data = pd.read_csv("./diffusion_llms/data/train.csv")

# Join words into a regex pattern
pattern = r'\b(?:' + '|'.join(map(re.escape, TRIGGER_WORDS)) + r')\b'

# Create a single column that's True if any word is present
data['is_trigger'] = data.user_prompt.str.contains(pattern, case=False, regex=True)

# Sample N random & N trigger words
prompts = {}
prompts["random"] = data[data.is_trigger].sample(N).user_prompt.to_list()
prompts["trigger"] = data[~data.is_trigger].sample(N).user_prompt.to_list()


print("Loading model...")
# instantiate the LladaBackbone model
model = LladaBackbone()

# Loop through and save distributions
print("Forwarding prompts...")
for group in ["random", "trigger"]:
    # Save
    probs_buffer = []
    for prompt in tqdm(prompts[group], desc=f"{group} prompts"):

        # Fix: use input_ids (plural) instead of input_id
        prompt = model.tokenizer(prompt, return_tensors="pt").input_ids

        # Create input with mask
        x = torch.full((1, seq_len), mask_id, dtype=torch.long).to(
            model.device
        )

        # Add prompt 
        x[:, : prompt.shape[1]] = prompt.clone()

        # Needed for forward
        dummy_target = torch.zeros_like(x)

        # Forward
        out = model(x, target=dummy_target)
        logits = out['logits']

        # Get logits corresponding to EOS
        # (logits, )
        logits = logits[:, :, model.tokenizer.eos_token_id].squeeze(0) 

        # Save probs
        probs = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
        
        # Convert to numpy
        probs = probs.detach().cpu().numpy().tolist()

        # Store
        probs_buffer.append(probs)

    # Save the result
    arr = np.array(probs_buffer)
    assert arr.shape == (N, seq_len)
    np.save(
        f"{group}_probs.npy",
        arr,
    )

print("Succesffully saved all probs!")

# Test
print("Loading logits back to plot...")
for group in ["random", "trigger"]:
    
    # Load back the probs
    arr = np.load(f"{group}_probs.npy")
    
    # Compute mean along the N dimension
    avg = np.mean(arr, axis = 0)
    assert len(avg) == seq_len

    # path to save
    path = f"{group}_eos.png"
    # plot the distribution of eos probabilities per position
    plt.figure(figsize=(20, 10))
    sns.set(style="whitegrid")
    sns.set_palette("pastel")
    plt.title(f"EOS Probability per Position ({group.capitalize()} Prompts)")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.xticks(np.arange(0, 1024, step=50))
    plt.yticks(np.arange(0, np.max(avg) + np.std(avg), step=0.1))
    plt.xlim(0, 1024)
    plt.ylim(0, 1)
    plt.plot(avg, color='blue', label='Eos probabilities')
    print(f"Successfully saved plot saved to {path}")

print("Done!")