import json
import os
import sys

import tiktoken
import torch
from diffusion_llms.models.gpt2_diffusion import DiffuGPT
from diffusion_llms.baseline.model_baseline import DistilBertClassifier
from diffusion_llms.utils import get_device
from transformers import AutoTokenizer


# From the command line we can specify the config.file
if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    print("No path/to/config.json provided, defaulting to './config.json'")
    CONFIG_PATH = "./config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

device = get_device()
# Get prompt
if config["user_prompt"]:
    prompt = config["user_prompt"]
else:
    prompt = "What is the capital of France?"

# Create DistilBERT classifier to predict the length of the answer
length_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
length_predictor_model = DistilBertClassifier(n_classes=5)
length_predictor_model.load_state_dict(torch.load("diffusion_llms/baseline/checkpoints/DistilBERT_1.pth"))
length_predictor_model.to(device)
length_predictor_model.eval()
steps = [32, 64, 128, 256, 512]

# Get the length of the answer
with torch.no_grad():
    input_enc = length_tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=512).to(device)
    logits, _ = length_predictor_model(input_enc['input_ids'], input_enc['attention_mask'])
    pred = torch.argmax(logits, dim=1)
    max_new_tokens = steps[pred.item()]
    print("MAX_NEW_TOKENS", max_new_tokens)


# Tokenize
tokenizer = tiktoken.get_encoding("gpt2")

# Mask token
mask_token = tokenizer.decode([config["mask_id"]])

# Tokenize the prompt
input_ids = torch.tensor(
        [50256] + tokenizer.encode(prompt)
    ).unsqueeze(0).to(device)

# Instantiate a model (new or pretrained)
if os.path.exists(config["init_from"]):
    model = DiffuGPT.from_pretrained(config["init_from"])
else:
    model = DiffuGPT(CONFIG_PATH)

model = model.to(device)
# Set evaluation mode
model.eval()

# Generate
n = config["n_samples"]
print(f"\nGenerating {n} samples...\n")
for _ in range(n):

    # List of tensors of shape (B, seq_len)
    xs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=config["temperature"],
        top_k=config["top_k"],
        denoising_strategy=config["denoising_strategy"],
        diffusion_steps=config["diffusion_steps"],
    )

    # Illustrate the diffusion process
    for x in xs:
        out = tokenizer.decode(x[0].tolist()).replace(mask_token, "<mask>")
        print(out)
        print()
    print("-" * 89)
    print(f"generated {max_new_tokens} tokens")
    print("-" * 89)
