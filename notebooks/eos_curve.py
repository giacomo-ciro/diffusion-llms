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
prompts["random"] = data[data.is_trigger].sample(N).to_list()
prompts["trigger"] = data[~data.is_trigger].sample(N).to_list()


print("Loading model...")
# instantiate the LladaBackbone model
model = LladaBackbone()

# Save
logits_buffer = []

# Loop through and save distributions
print("Forwarding prompts")
for group in ["random", "trigger"]:
    for prompt in tqdm(prompts[group].values(), desc=f"{group} prompts"):

        # Fix: use input_ids (plural) instead of input_id
        prompt = model.tokenizer(prompt, return_tensors="pt").input_ids

        # Create input with mask
        x = torch.full((1, 1024), mask_id, dtype=torch.long).to(
            model.device
        )

        # Add prompt 
        x[:, : prompt.shape[1]] = prompt.clone()

        # Forward
        logits = model(x)

        # Get logits corresponding to EOS
        logits = logits[:, :, model.tokenizer.eos_token_id].squeeze(0)  # (1024)

        # Store
        logits_buffer.append(logits)

    # Save the result
    np.save(
        f"{group}_logits.npy",
        np.array(logits),
    )

print("Succesffully saved all logits!")

# Test
print(f"Loading logits back...")
for group in ["random", "trigger"]:
    arr = np.load(f"{group}_logits.npy")
print("Successfully loaded back all logits!")

print("Done")
#     p = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)

# # plot the distribution of eos probabilities per position
# p = p.cpu().numpy()
# plt.figure(figsize=(20, 10))
# sns.set(style="whitegrid")
# sns.set_palette("pastel")
# plt.title("Eos probabilities per position")
# plt.xlabel("Position")
# plt.ylabel("Probability")
# plt.xticks(np.arange(0, 1024, step=50))
# plt.yticks(np.arange(0, np.max(p) + np.std(p), step=0.1))
# plt.xlim(0, 1024)
# plt.ylim(0, 1)
# plt.plot(p, color='blue', label='Eos probabilities')