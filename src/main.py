import torch
from model import GPT2

model = GPT2()

print("Successfully loaded GPT-2")

# Test model Logic
# 1 sentence of len 10, from a vocab of 100 toks
T = 10
sentence = torch.randint(0, 100, size=(1, T+1))
X = sentence[:, :-1]
y = sentence[:, 1:]

out = model.step(
    batch = (X, y),     # read in as idx, targets = batch
    batch_idx = 0  
)

# The loss
print(out)

# TODO: 
# 1. Implement tokenizer (inside lightning module)
# 2. Set up lightning training