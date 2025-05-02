# evaluation/eval_eos.py

import torch
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm

# Add parent directory to sys.path to allow imports from sibling directories
# Assumes eval_eos.py is in diffusion-llms/evaluation/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import GPT2  # Import from parent directory
from datamodule import MemmapDataModule # Import from parent directory
from utils import get_annealing_mask # Import from parent directory

def evaluate_first_step_eos(config_path: str, ckpt_path: str, device: str):
    """
    Evaluates the model's ability to predict the EOS token at the correct position
    during the first step of a simulated diffusion process (all future tokens masked).

    Args:
        config_path (str): Path to the config.json file corresponding to the checkpoint.
        ckpt_path (str): Path to the model checkpoint (.ckpt) file.
        device (str): Device to run evaluation on ('cuda' or 'cpu').
    """

    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"Using device: {device}")

    # --- Load Config ---
    if not os.path.exists(config_path):
        print(f"[!] Error: Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = json.load(f)

    # Essential parameters for evaluation
    eos_token_id = config.get("eos_token_id")
    pad_token_id = config.get("pad_token_id")
    mask_id = config.get("mask_id")
    context_length = config.get("context_length")
    memmap_path_for_eval = config.get("memmap_path") # Use the same dataset split logic

    if any(v is None for v in [eos_token_id, pad_token_id, mask_id, context_length, memmap_path_for_eval]):
        print("[!] Error: One or more required parameters (eos_token_id, pad_token_id, mask_id, context_length, memmap_path) not found in config.")
        sys.exit(1)

    # Ensure the dataset used for evaluation is intended for variable length
    if not config.get("padded_dataset", False):
         print("[!] Warning: Config 'padded_dataset' is not set to true. Evaluation assumes data is in [text+eos+pad...] format.")
         # Proceeding, but the dataset MUST have been created with prepare_var_len.py

    print(f"EOS: {eos_token_id}, PAD: {pad_token_id}, MASK: {mask_id}, Context: {context_length}")

    # --- Load Model ---
    if not os.path.exists(ckpt_path):
        print(f"[!] Error: Checkpoint file not found at {ckpt_path}")
        sys.exit(1)
    try:
        # Use the class method to load, passing the specific config path
        model = GPT2.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            config_path=config_path # Crucial: ensure model structure matches config
        )
        model.eval()
        model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading model checkpoint: {e}")
        sys.exit(1)

    # --- Load Data ---
    print("Setting up datamodule...")
    try:
        # Use the same config for the datamodule to ensure consistent splitting
        data_module = MemmapDataModule(config_path=config_path)
        data_module.setup('test') # Prepare train/val/test splits
        test_dataloader = data_module.test_dataloader()
        print(f"Test dataset size (batches): {len(test_dataloader)}")
        if len(test_dataloader) == 0:
             print("[!] Error: Test dataloader is empty. Check dataset path and val_test_perc in config.")
             sys.exit(1)
    except Exception as e:
        print(f"[!] Error setting up datamodule or dataloader: {e}")
        sys.exit(1)

    # --- Evaluation Loop ---
    all_eos_probs = []
    all_is_correct = []
    all_eos_ranks = []
    skipped_samples = 0

    print("Starting evaluation...")
    pbar = tqdm(test_dataloader, desc="Evaluating Batches")
    for batch in pbar:
        # Batch contains (X, y, mask_for_training)
        # We only need X for this evaluation. X has shape [B, context_length]
        # X should represent sequences like [t1, ..., t_prompt_len, eos, pad, ...]
        input_sequences = batch[0].to(device)
        B = input_sequences.size(0)

        for i in range(B):
            full_sequence = input_sequences[i] # Shape: [context_length]

            # Find the ground truth position of the EOS token
            eos_indices = (full_sequence == eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_indices) == 0:
                # This shouldn't happen if data was prepared correctly by prepare_var_len.py
                # print(f"Warning: EOS token {eos_token_id} not found in sequence, skipping.")
                skipped_samples += 1
                continue

            eos_pos_gt = eos_indices[0].item() # Use the first occurrence
            prompt_len = eos_pos_gt # Index of EOS is the length of the actual text prompt

            # Skip if EOS is the very first token (cannot predict)
            if prompt_len == 0:
                skipped_samples += 1
                continue

            # --- Simulate First Diffusion Step ---
            # 1. Create input_ids: [prompt tokens, mask_id, ..., mask_id]
            input_ids_eval = full_sequence.clone().unsqueeze(0) # Add batch dim [1, context_length]
            input_ids_eval[0, prompt_len:] = mask_id

            # 2. Create attention mask: Full attention (p=1.0)
            # Allows all prompt tokens to attend to each other, and masked tokens to attend to prompt
            attention_mask_eval = get_annealing_mask(context_length, B=1, p=1.0).to(device)

            # 3. Create input mask for the forward pass: True where input_ids_eval has mask_id
            input_mask_for_forward_eval = (input_ids_eval == mask_id)

            # --- Forward Pass ---
            with torch.no_grad():
                # Targets are not used for loss calculation here, but needed by signature
                # Pass dummy or actual sequence; input_mask ensures loss isn't computed anyway if needed
                logits, _ = model.forward(
                    input_ids=input_ids_eval,
                    targets=input_ids_eval, # Dummy target
                    input_mask=input_mask_for_forward_eval, # Mask for *input* tokens
                    attention_mask=attention_mask_eval
                )
                # logits shape: [1, context_length, vocab_size]

            # --- Extract Prediction for EOS Position ---
            # Logits at index (prompt_len - 1) predict the token at index prompt_len (the EOS position)
            logits_for_eos_pos = logits[0, prompt_len - 1, :] # Shape: [vocab_size]
            probs_for_eos_pos = torch.softmax(logits_for_eos_pos, dim=-1)

            # --- Calculate Metrics for this sample ---
            prob_of_eos = probs_for_eos_pos[eos_token_id].item()
            predicted_token_id = torch.argmax(probs_for_eos_pos).item()

            # Calculate rank
            sorted_indices = torch.argsort(probs_for_eos_pos, descending=True)
            # Find the position (rank) of the true eos_token_id in the sorted list
            rank_tensor = (sorted_indices == eos_token_id).nonzero(as_tuple=True)[0]
            # Add 1 because rank is 1-based
            rank_of_eos = rank_tensor.item() + 1 if len(rank_tensor) > 0 else float('inf') # Handle case where EOS is somehow not in vocab (shouldn't happen)

            all_eos_probs.append(prob_of_eos)
            all_is_correct.append(1 if predicted_token_id == eos_token_id else 0)
            if rank_of_eos != float('inf'):
                all_eos_ranks.append(rank_of_eos)

    # --- Aggregate and Report Results ---
    num_evaluated = len(all_eos_probs)
    num_ranks_calculated = len(all_eos_ranks)

    print("\n--- Evaluation Complete ---")
    print(f"Total samples processed: {num_evaluated + skipped_samples}")
    print(f"Samples skipped (e.g., no EOS found, prompt_len=0): {skipped_samples}")
    print(f"Samples evaluated: {num_evaluated}")

    if num_evaluated > 0:
        mean_eos_prob = np.mean(all_eos_probs)
        top_1_accuracy = np.mean(all_is_correct)

        print(f"\n--- First-Step EOS Prediction Metrics ---")
        print(f"Mean EOS Probability at GT Position: {mean_eos_prob:.4f}")
        print(f"Top-1 EOS Accuracy at GT Position:   {top_1_accuracy:.4f}")

        if num_ranks_calculated > 0:
             # Calculate MRR only on samples where rank could be determined
             mrr = np.mean([1.0 / r for r in all_eos_ranks])
             print(f"Mean Reciprocal Rank (MRR) of EOS:  {mrr:.4f} (calculated on {num_ranks_calculated} samples)")
        else:
             print("Mean Reciprocal Rank (MRR) of EOS: Could not be calculated (no valid ranks).")

    else:
        print("\nNo valid samples were evaluated. Cannot calculate metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate First-Step EOS Prediction of a Diffusion LLM.")
    parser.add_argument("config_path", type=str, help="Path to the config.json file used for training the model.")
    parser.add_argument("ckpt_path", type=str, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()

    evaluate_first_step_eos(args.config_path, args.ckpt_path, args.device)
