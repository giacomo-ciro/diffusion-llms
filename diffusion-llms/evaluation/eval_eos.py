# evaluation/eval_eos.py

import torch
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add parent directory to sys.path to allow imports from sibling directories
# Assumes eval_eos.py is in diffusion-llms/evaluation/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import GPT2  # Import from parent directory
from utils import get_annealing_mask # Import from parent directory

# --- Direct Dataset Class (Simplified from MemmapTokenDataset) ---
# We need this to load the test data directly without splitting logic
class DirectMemmapTestDataset(Dataset):
    """Loads data directly from a specified memmap file for testing."""
    def __init__(self, memmap_path: str, context_length: int, config_dir: str):
        # Resolve potential relative paths from config file location
        # If path is not absolute, join it with the config file's directory
        if not os.path.isabs(memmap_path):
            memmap_path = os.path.join(config_dir, memmap_path)
        self.memmap_path = os.path.abspath(memmap_path) # Ensure absolute path

        if not os.path.exists(self.memmap_path):
             raise FileNotFoundError(f"Memmap file not found: {self.memmap_path}")

        self.data = np.memmap(self.memmap_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
        # Calculate effective length assuming stride is context_length
        self.effective_length = len(self.data) // self.context_length
        print(f"DirectMemmapTestDataset: Loaded {self.memmap_path}, effective length: {self.effective_length}")


    def __len__(self):
        return self.effective_length

    def __getitem__(self, idx):
        if idx >= self.effective_length:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.effective_length} samples.")

        # Calculate start index assuming data is stored in blocks
        start_idx = idx * self.context_length
        # Get sequence of indices of shape (context_length,)
        X = self.data[start_idx : start_idx + self.context_length].copy()
        X = torch.from_numpy(X).to(torch.int64)

        # Return only X, as we don't need y or mask for this specific eval
        return X

# --- Evaluation Function ---
def evaluate_first_step_eos(config: dict, device: str):
    """
    Evaluates the model's ability to predict the EOS token at the correct position
    during the first step of a simulated diffusion process (all future tokens masked).

    Args:
        config (dict): The loaded configuration dictionary.
        device (str): The device to run evaluation on ('cuda', 'mps', or 'cpu').
    """

    # --- Get parameters from config ---
    ckpt_path_rel = config.get("ckpt_path_for_eval")
    test_data_path_rel = config.get("test_data_path_for_eval")
    eval_batch_size = config.get("eval_batch_size", 8) # Default to 8 if not set
    eos_token_id = config.get("eos_token_id")
    pad_token_id = config.get("pad_token_id")
    mask_id = config.get("mask_id")
    context_length = config.get("context_length")
    config_path_origin = config.get("_config_path_origin") # Path where config was loaded from

    # --- Validate required parameters ---
    if any(v is None for v in [ckpt_path_rel, test_data_path_rel, eos_token_id, mask_id, context_length]):
        print("[!] Error: One or more required parameters (ckpt_path_for_eval, test_data_path_for_eval, eos_token_id, mask_id, context_length) not found in config.")
        sys.exit(1)
    if "UPDATE_THIS_PATH" in ckpt_path_rel:
        print(f"[!] Error: Please update 'ckpt_path_for_eval' in the config file ({config_path_origin}) with the actual path to your .ckpt file.")
        sys.exit(1)

    # --- Resolve paths relative to the config file's location ---
    if config_path_origin:
        config_dir = os.path.dirname(config_path_origin)
        # If the path is not absolute, join it with the config directory
        ckpt_path = os.path.join(config_dir, ckpt_path_rel) if not os.path.isabs(ckpt_path_rel) else ckpt_path_rel
        test_data_path = os.path.join(config_dir, test_data_path_rel) if not os.path.isabs(test_data_path_rel) else test_data_path_rel
        # Make paths absolute for clarity and robustness
        ckpt_path = os.path.abspath(ckpt_path)
        test_data_path = os.path.abspath(test_data_path)
    else:
        # Fallback if origin path wasn't stored (shouldn't happen with current main block)
        ckpt_path = os.path.abspath(ckpt_path_rel)
        test_data_path = os.path.abspath(test_data_path_rel)
        print("[!] Warning: Config origin path not found, resolving paths relative to current working directory.")


    print(f"Using evaluation config: {config_path_origin or 'N/A'}")
    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"Loading TEST data from: {test_data_path}")
    print(f"Using device: {device}")
    print(f"Evaluation Batch Size: {eval_batch_size}")
    print(f"EOS: {eos_token_id}, PAD: {pad_token_id}, MASK: {mask_id}, Context: {context_length}")


    # --- Load Model ---
    if not os.path.exists(ckpt_path):
        print(f"[!] Error: Checkpoint file not found at {ckpt_path}")
        sys.exit(1)
    try:
        # Pass the original config path used for training to load_from_checkpoint
        # This config path is needed by the GPT2 class to know its own architecture etc.
        model = GPT2.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            config_path=config_path_origin # Use the path of the loaded config
        )
        model.eval()
        model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading model checkpoint: {e}")
        print(f"Ensure the config file ({config_path_origin}) matches the structure expected by the checkpoint.")
        sys.exit(1)

    # --- Load Specific Test Data ---
    print("Setting up test dataloader...")
    try:
        # Pass the resolved absolute path and the config directory
        test_dataset = DirectMemmapTestDataset(test_data_path, context_length, os.path.dirname(config_path_origin or '.'))
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size, # Use eval_batch_size from config
            shuffle=False,
            num_workers=2, # Adjust as needed
            pin_memory=True if device == 'cuda' else False
        )
        print(f"Test dataset size (samples): {len(test_dataset)}")
        print(f"Test dataloader size (batches): {len(test_dataloader)}")
        if len(test_dataloader) == 0:
             print("[!] Error: Test dataloader is empty. Check test data path.")
             sys.exit(1)
    except Exception as e:
        print(f"[!] Error setting up test dataset or dataloader: {e}")
        sys.exit(1)

    # --- Evaluation Loop ---
    all_eos_probs = []
    all_is_correct = []
    all_eos_ranks = []
    skipped_samples = 0

    print("Starting evaluation...")
    pbar = tqdm(test_dataloader, desc="Evaluating Batches")
    # The dataloader now yields batches of X directly
    for input_sequences in pbar:
        input_sequences = input_sequences.to(device) # Shape [B, context_length]
        B = input_sequences.size(0)

        for i in range(B):
            full_sequence = input_sequences[i] # Shape: [context_length]

            # Find the ground truth position of the EOS token
            eos_indices = (full_sequence == eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_indices) == 0:
                skipped_samples += 1
                continue

            eos_pos_gt = eos_indices[0].item()
            prompt_len = eos_pos_gt

            if prompt_len == 0:
                skipped_samples += 1
                continue

            # --- Simulate First Diffusion Step ---
            input_ids_eval = full_sequence.clone().unsqueeze(0)
            input_ids_eval[0, prompt_len:] = mask_id

            attention_mask_eval = get_annealing_mask(context_length, B=1, p=1.0).to(device)
            input_mask_for_forward_eval = (input_ids_eval == mask_id)

            # --- Forward Pass ---
            with torch.no_grad():
                logits, _ = model.forward(
                    input_ids=input_ids_eval,
                    targets=input_ids_eval, # Dummy target
                    input_mask=input_mask_for_forward_eval, # Mask for *input* tokens
                    attention_mask=attention_mask_eval
                )

            # --- Extract Prediction for EOS Position ---
            logits_for_eos_pos = logits[0, prompt_len - 1, :]
            probs_for_eos_pos = torch.softmax(logits_for_eos_pos, dim=-1)

            # --- Calculate Metrics for this sample ---
            prob_of_eos = probs_for_eos_pos[eos_token_id].item()
            predicted_token_id = torch.argmax(probs_for_eos_pos).item()

            sorted_indices = torch.argsort(probs_for_eos_pos, descending=True)
            rank_tensor = (sorted_indices == eos_token_id).nonzero(as_tuple=True)[0]
            rank_of_eos = rank_tensor.item() + 1 if len(rank_tensor) > 0 else float('inf')

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
             mrr = np.mean([1.0 / r for r in all_eos_ranks])
             print(f"Mean Reciprocal Rank (MRR) of EOS:  {mrr:.4f} (calculated on {num_ranks_calculated} samples)")
        else:
             print("Mean Reciprocal Rank (MRR) of EOS: Could not be calculated (no valid ranks).")
    else:
        print("\nNo valid samples were evaluated. Cannot calculate metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate First-Step EOS Prediction of a Diffusion LLM.")
    # Only argument is the path to the config file
    parser.add_argument("eval_config_path", type=str, help="Path to the .json config file containing evaluation parameters (ckpt path, test data path, etc.).")

    args = parser.parse_args()

    # --- Load the Evaluation Config ---
    # Get absolute path for robustness
    eval_config_path_abs = os.path.abspath(args.eval_config_path)
    if not os.path.exists(eval_config_path_abs):
        print(f"[!] Error: Evaluation config file not found at {eval_config_path_abs}")
        sys.exit(1)
    try:
        with open(eval_config_path_abs, "r") as f:
            eval_config = json.load(f)
            # Store the original path for resolving relative paths later
            eval_config["_config_path_origin"] = eval_config_path_abs
    except Exception as e:
        print(f"[!] Error loading or parsing config file {eval_config_path_abs}: {e}")
        sys.exit(1)


    # --- Auto-detect Device ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available(): # Check for MPS (Apple Silicon GPU)
        device = "mps"
    else:
        device = "cpu"

    # --- Run Evaluation ---
    evaluate_first_step_eos(eval_config, device)
