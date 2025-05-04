import torch
import os
import sys
import argparse
from tqdm import tqdm
import tiktoken
import numpy as np
import json
import time

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import GPT2
from utils import compute_binary_metrics
from datamodule import MemmapTokenDataset

def main(model, path_to_test, ans):
    model.config["memmap_path"] = path_to_test
    assert os.path.exists(model.config["memmap_path"]), "No test dataset found at " + model.config["memmap_path"]
    
    ds = MemmapTokenDataset(model.config)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Get params from the model
    pad_token_id = model.config["pad_token_id"]
    eos_token_id = model.config["eos_token_id"]
    mask_id = model.config["mask_id"]

    # Evaluate
    perces = [0, 0.25, 0.50, 0.75]
    for perc in range(len(perces)):
        if perces[perc] == 0:
            continue
        print(f"Masking a random token in the {perc+1}-th quarter of the sentence")
        accuracies, recalls, precisions, f1s = [],[],[],[]
        for sample in tqdm(ds):

            # Read batch
            input_ids, targets, _ = sample
            
            # Get length
            seq_len = len(input_ids)

            # Sample random index
            idx = torch.randint(
                int(perces[perc-1] * len(input_ids)),
                int(perces[perc] * len(input_ids)),
                size=(1,),
                dtype = int
            ).item()

            # Predict remaining ones
            max_new_tokens = seq_len - idx
            
            # Predict 4 tokens at each step
            diffusion_steps = max_new_tokens // 4

            # Remove the end and predict
            generated_ids = model.generate(
                input_ids[:idx].unsqueeze(0).to(device),
                max_new_tokens = max_new_tokens,
                diffusion_steps = diffusion_steps,
                pipeline = "diffusion",
                denoising_strategy = "entropy",
                temperature = 1.0,
                top_k = None
            )[-1].view(-1).tolist()
            
            # Get prediction to compute binary metrics (1 if pad, 0 otherwise)
            tp, tn, fp, fn = 0, 0, 0, 0
            for i, pred in enumerate(generated_ids[idx:]):
                true = input_ids[idx+i]
                if pred == pad_token_id and true == pad_token_id:
                    tp += 1
                elif pred == pad_token_id and true != pad_token_id:
                    fp += 1
                elif pred != pad_token_id and true == pad_token_id:
                    fn += 1
                else:
                    tn += 1
            assert tp+fp+tn+fn == len(generated_ids[idx:])

            # Compute binary metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Store to compute mean
            accuracies.append(accuracy)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

            # # The sentence fed to the model
            # input_sentence = tokenizer.decode(input_ids[:idx].tolist())
            # print(f"\nInput:\n{input_sentence}")

            # # What the model predicted
            # generated_completion = tokenizer.decode([i if i <= eos_token_id else eos_token_id for i in generated_ids[idx:]])
            # generated_completion = generated_completion.replace("<|endoftext|>","<eos>")
            # print(f"Pred:\n{generated_completion}")

            # # The true continuation
            # true_completion = tokenizer.decode([i if i <= eos_token_id else eos_token_id for i in input_ids[idx:]])
            # true_completion = true_completion.replace("<|endoftext|>","<eos>")
            # print(f"True:\n{true_completion}")

            # if input("Press enter to sample another: ") != "":
            #     break
            if len(accuracies) >= 100:
                break
        print("Computing average metrics...")
        ans[perc] = {
            "accuracy" : np.mean(accuracies),
            "recall" : np.mean(recalls),
            "precision" : np.mean(precisions),
            "f1" : np.mean(f1s)
        }
    
    with open(f"{int(time.time())}.json", "w") as f:
        json.dump(ans, f)
    
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path/to/model.ckpt")
    args = parser.parse_args()

    # Load config
    assert os.path.exists(args.config)
    with open(args.config, "r") as f:
        config = json.load(f)

    # Load model
    if os.path.exists(config["init_from"]):
        model = GPT2.from_pretrained(config["init_from"])
    else:
        assert config["pipeline"] == "diffusion"
        model = GPT2(args.config)

    # Get device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # get test data
    path_to_test = config["memmap_path"]

    # To save the results of the test
    ans = {
        "model": config["init_from"]
    }
    
    # Run test
    main(model, path_to_test, ans)