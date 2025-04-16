# eval_eos_prediction.py
import sys
import torch
import json
import tiktoken
import numpy as np
from model import GPT2
from utils import get_annealing_mask

# Load config
if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    CONFIG_PATH = './local_config.json'

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Tokenizer setup
enc = tiktoken.get_encoding("gpt2")
eos_token_id = config.get("eos_token_id", enc.eot_token)
pad_token_id = config.get("pad_token_id", 50256)

def encode(x):
    return enc.encode(x, allowed_special={"<|endoftext|>"})

def decode(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy().tolist()
    return enc.decode(x)

# Load model
print(f"Loading model from {CONFIG_PATH}...")
model = GPT2(CONFIG_PATH)
model.eval()

# Disable logging
original_log_method = model.log
model.log = lambda *args, **kwargs: None

# Test prompts
test_prompts = [
    "Hello, what's your name?",
    "Write a short poem about",
    "The capital of France is",
    "I enjoy walking in the park because"
]

def test_eos_prediction(model, prompt, eos_token_id, num_positions=10):
    """
    Test EOS prediction at the first diffusion step for a specific prompt.
    """
    prompt_tokens = encode(prompt)
    B = 1  # batch size
    prompt_len = len(prompt_tokens)
    seq_len = prompt_len + num_positions
    
    # Create input with prompt followed by masks
    input_ids = torch.full((B, seq_len), config["mask_id"], dtype=torch.long)
    input_ids[0, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long)
    
    # Create mask for prediction (only predict masked positions)
    input_mask = torch.zeros((B, seq_len), dtype=torch.bool)
    input_mask[0, prompt_len:] = True
    
    # Full attention for testing
    attn_mask = get_annealing_mask(seq_len, B, 1.0)
    
    # Run prediction 
    with torch.no_grad():
        logits, _ = model.forward(input_ids, input_ids, input_mask, attn_mask)
    
    # Get token predictions
    probs = torch.softmax(logits, dim=-1)
    
    # Get EOS probabilities at each masked position
    eos_probs = probs[0, prompt_len:, eos_token_id].cpu().numpy()
    
    # Get top token predictions
    top_tokens = torch.argmax(logits[0, prompt_len:], dim=-1).cpu().numpy()
    top_probs = torch.max(probs[0, prompt_len:], dim=-1)[0].cpu().numpy()
    
    return {
        "eos_probs": eos_probs,
        "top_tokens": top_tokens,
        "top_probs": top_probs,
        "has_eos": eos_token_id in top_tokens,
        "eos_positions": np.where(top_tokens == eos_token_id)[0],
        "max_eos_prob": np.max(eos_probs),
        "max_eos_position": np.argmax(eos_probs) if len(eos_probs) > 0 else -1
    }

print("\nEvaluating EOS prediction capabilities...\n")
print(f"EOS token ID: {eos_token_id}, Pad token ID: {pad_token_id}")

overall_results = []
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    results = test_eos_prediction(model, prompt, eos_token_id)
    
    print(f"Max EOS probability: {results['max_eos_prob']:.4f} at position {results['max_eos_position'] + 1}")
    print(f"EOS appears in top predictions: {results['has_eos']}")
    
    if results['eos_positions'].size > 0:
        positions = results['eos_positions'] + 1  # Convert to 1-indexed positions
        print(f"EOS predicted at positions: {positions}")
    else:
        print("EOS not predicted in any position")
    
    # Visualize token probabilities
    print("\nPosition | Top Token | Prob  | EOS Prob")
    print("-" * 40)
    for i in range(len(results['top_tokens'])):
        token_text = decode([results['top_tokens'][i]])
        if len(token_text) > 10:
            token_text = token_text[:10] + "..."
        print(f"{i+1:8d} | {token_text:9s} | {results['top_probs'][i]:.4f} | {results['eos_probs'][i]:.4f}")
    
    overall_results.append(results)

# Compute average metrics
avg_max_eos_prob = sum(r['max_eos_prob'] for r in overall_results) / len(overall_results)
eos_prediction_rate = sum(1 for r in overall_results if r['has_eos']) / len(overall_results)

print(f"\nOverall Results:")
print(f"Average max EOS probability: {avg_max_eos_prob:.4f}")
print(f"EOS prediction rate: {eos_prediction_rate:.2f}")

# Restore original log method
model.log = original_log_method