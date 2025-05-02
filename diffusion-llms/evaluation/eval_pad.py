import torch
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import wandb # Import wandb
import tiktoken
from sklearn.metrics import precision_recall_fscore_support, accuracy_score # For metrics

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import GPT2
from utils import get_annealing_mask # Although not directly used for generation logic here

# --- Direct Dataset Class ---
class DirectMemmapTestDataset(Dataset):
    """Loads data directly from a specified memmap file for testing."""
    def __init__(self, memmap_path: str, context_length: int, config_dir: str):
        if not os.path.isabs(memmap_path):
            memmap_path = os.path.join(config_dir, memmap_path)
        self.memmap_path = os.path.abspath(memmap_path)
        if not os.path.exists(self.memmap_path):
             raise FileNotFoundError(f"Memmap file not found: {self.memmap_path}")
        self.data = np.memmap(self.memmap_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
        self.effective_length = len(self.data) // self.context_length
        print(f"DirectMemmapTestDataset: Loaded {self.memmap_path}, effective length: {self.effective_length}")

    def __len__(self):
        return self.effective_length

    def __getitem__(self, idx):
        if idx >= self.effective_length:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.effective_length} samples.")
        start_idx = idx * self.context_length
        X = self.data[start_idx : start_idx + self.context_length].copy()
        X_tensor = torch.from_numpy(X).to(torch.int64)
        # Also return the numpy array for easier ground truth handling
        return X_tensor, X

# --- Evaluation Function ---
def evaluate_final_padding(config: dict, device: str):
    """
    Evaluates the model's ability to generate correct padding after the
    end-of-sentence token in a full diffusion generation process.

    Args:
        config (dict): The loaded configuration dictionary.
        device (str): The device to run evaluation on ('cuda', 'mps', or 'cpu').
    """

    # --- Get parameters from config ---
    ckpt_path_rel = config.get("ckpt_path_for_eval")
    test_data_path_rel = config.get("test_data_path_for_eval")
    eval_batch_size = config.get("eval_batch_size", 8) # Use smaller batch for generation
    eos_token_id = config.get("eos_token_id")
    pad_token_id = config.get("pad_token_id")
    mask_id = config.get("mask_id") # Needed by model class, not directly here
    context_length = config.get("context_length")
    config_path_origin = config.get("_config_path_origin")
    # Generation parameters from config (important now!)
    diffusion_steps = config.get("diffusion_steps", 4)
    temperature = config.get("temperature", 1.0)
    top_k = config.get("top_k", None)
    denoising_strategy = config.get("denoising_strategy", "random")
    # Wandb parameters
    log_to_wandb = config.get("wandb_eval", False)
    wandb_project = config.get("project_name", "diffusion-llms-eval")
    wandb_run_name_base = config.get("run_name", "eval-padding")
    wandb_run_name_suffix = config.get("wandb_eval_run_name_suffix", "-eval-padding") # Different suffix
    wandb_run_name = f"{wandb_run_name_base}{wandb_run_name_suffix}"


    # --- Validate required parameters ---
    if any(v is None for v in [ckpt_path_rel, test_data_path_rel, eos_token_id, pad_token_id, mask_id, context_length]):
        print("[!] Error: One or more required parameters not found in config.")
        sys.exit(1)
    if "UPDATE_THIS" in ckpt_path_rel:
        print(f"[!] Error: Please update 'ckpt_path_for_eval' in the config file ({config_path_origin}).")
        sys.exit(1)

    # --- Resolve paths ---
    config_dir = os.path.dirname(config_path_origin or '.')
    ckpt_path = os.path.join(config_dir, ckpt_path_rel) if not os.path.isabs(ckpt_path_rel) else ckpt_path_rel
    test_data_path = os.path.join(config_dir, test_data_path_rel) if not os.path.isabs(test_data_path_rel) else test_data_path_rel
    ckpt_path = os.path.abspath(ckpt_path)
    test_data_path = os.path.abspath(test_data_path)

    print(f"Using evaluation config: {config_path_origin or 'N/A'}")
    print(f"Loading checkpoint from: {ckpt_path}")
    print(f"Loading TEST data from: {test_data_path}")
    print(f"Using device: {device}")
    print(f"Evaluation Batch Size (for generation): {eval_batch_size}")
    print(f"EOS: {eos_token_id}, PAD: {pad_token_id}, Context: {context_length}")
    print(f"Generation settings: steps={diffusion_steps}, temp={temperature}, top_k={top_k}, strategy='{denoising_strategy}'")

    # --- Initialize Tokenizer ---
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        eos_token_str = tokenizer.decode([eos_token_id])
        pad_token_str = "[PAD]" # Assign visually distinct string
    except Exception as e:
        print(f"Warning: Could not initialize tokenizer: {e}.")
        tokenizer = None
        eos_token_str = f"ID:{eos_token_id}"
        pad_token_str = f"ID:{pad_token_id}"


    # --- Initialize WandB ---
    if log_to_wandb:
        try:
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            print(f"Logging evaluation results to WandB project '{wandb_project}', run '{wandb_run_name}'")
        except Exception as e:
            print(f"[!] Warning: Failed to initialize WandB: {e}. Proceeding without logging.")
            log_to_wandb = False
    else:
        print("WandB logging disabled.")

    # --- Load Model ---
    if not os.path.exists(ckpt_path):
        print(f"[!] Error: Checkpoint file not found at {ckpt_path}")
        if log_to_wandb: wandb.finish(exit_code=1)
        sys.exit(1)
    try:
        model = GPT2.load_from_checkpoint(checkpoint_path=ckpt_path, config_path=config_path_origin)
        model.eval()
        model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading model checkpoint: {e}")
        if log_to_wandb: wandb.finish(exit_code=1)
        sys.exit(1)

    # --- Load Specific Test Data ---
    print("Setting up test dataloader...")
    try:
        # NOTE: Generation is often done one sample at a time. Adjust batch_size if needed.
        # Using eval_batch_size, but might need to reduce if generation is slow/memory intensive.
        test_dataset = DirectMemmapTestDataset(test_data_path, context_length, config_dir)
        test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0) # num_workers=0 might be safer for generation loops
        print(f"Test dataset size (samples): {len(test_dataset)}")
        print(f"Test dataloader size (batches): {len(test_dataloader)}")
        if len(test_dataloader) == 0:
             print("[!] Error: Test dataloader is empty.")
             if log_to_wandb: wandb.finish(exit_code=1)
             sys.exit(1)
    except Exception as e:
        print(f"[!] Error setting up test dataset or dataloader: {e}")
        if log_to_wandb: wandb.finish(exit_code=1)
        sys.exit(1)

    # --- Evaluation Loop ---
    all_gt_pads = []
    all_pred_pads = []
    all_length_diffs = []
    skipped_samples = 0

    print("Starting evaluation (generating full sequences)...")
    pbar = tqdm(test_dataloader, desc="Evaluating Batches")
    # Dataloader yields (X_tensor, X_numpy)
    for batch_X_tensor, batch_X_numpy in pbar:
        batch_X_tensor = batch_X_tensor.to(device)
        B = batch_X_tensor.size(0)

        for i in range(B):
            # --- Prepare Ground Truth and Prompt ---
            gt_sequence_np = batch_X_numpy[i] # Full ground truth sequence as numpy
            eos_indices = np.where(gt_sequence_np == eos_token_id)[0]

            if len(eos_indices) == 0:
                skipped_samples += 1
                continue
            eos_pos_gt = eos_indices[0]
            prompt_len = eos_pos_gt # Length of the prompt part
            gt_len = prompt_len + 1 # Ground truth sequence length (including EOS)

            # Extract prompt tensor for generation
            prompt_tensor = batch_X_tensor[i, :prompt_len].unsqueeze(0).to(device) # Shape [1, prompt_len]
            if prompt_len == 0: # Handle empty prompt case if necessary
                 skipped_samples += 1
                 continue # Or generate from scratch if desired

            # --- Generate Full Sequence ---
            with torch.no_grad():
                # Use the model's generate method with diffusion pipeline
                # max_new_tokens should be context_length - prompt_len to fill the sequence
                max_new = context_length - prompt_len
                if max_new <= 0: # Should not happen with prepare_var_len data
                     skipped_samples += 1
                     continue

                # The generate method returns a list of steps, we need the last one
                generated_steps = model.generate(
                    input_ids=prompt_tensor,
                    pipeline="diffusion",
                    max_new_tokens=max_new,
                    temperature=temperature,
                    top_k=top_k,
                    denoising_strategy=denoising_strategy,
                    diffusion_steps=diffusion_steps
                    # Add other relevant generation params if needed from config
                )
                final_generated_tensor = generated_steps[-1][0] # Get final sequence, remove batch dim
                final_generated_np = final_generated_tensor.cpu().numpy()

            # --- Compare Padding ---
            # Define the "padding zone" - indices AFTER the ground truth EOS position
            padding_zone_start = gt_len
            padding_zone_end = context_length

            # Get ground truth tokens in the padding zone (should all be PAD)
            gt_padding_tokens = gt_sequence_np[padding_zone_start:padding_zone_end]

            # Get predicted tokens in the padding zone
            pred_padding_tokens = final_generated_np[padding_zone_start:padding_zone_end]

            # Store flattened lists of 0s (non-PAD) and 1s (PAD) for metrics
            # Treat PAD as the "positive" class
            # Make sure we're using numpy arrays with astype method, not PyTorch tensors
            if isinstance(gt_padding_tokens, torch.Tensor):
                gt_padding_tokens = gt_padding_tokens.cpu().numpy()
            if isinstance(pred_padding_tokens, torch.Tensor):
                pred_padding_tokens = pred_padding_tokens.cpu().numpy()
            
            gt_is_pad = (gt_padding_tokens == pad_token_id).astype(int)
            pred_is_pad = (pred_padding_tokens == pad_token_id).astype(int)

            all_gt_pads.extend(gt_is_pad.tolist())
            all_pred_pads.extend(pred_is_pad.tolist())

            # --- (Optional) Calculate Length Difference ---
            # Find first predicted PAD token index
            pred_pad_indices = np.where(final_generated_np[prompt_len:] == pad_token_id)[0]
            if len(pred_pad_indices) > 0:
                # Add prompt_len back to get index in full sequence
                first_pred_pad_idx = pred_pad_indices[0] + prompt_len
                # Generated length is the index of the first pad
                pred_len = first_pred_pad_idx
            else:
                # No PAD predicted, assume full length generated
                pred_len = context_length
            all_length_diffs.append(pred_len - gt_len)


    # --- Aggregate and Report Results ---
    num_evaluated = len(all_length_diffs) # Based on successful generations

    print("\n--- Evaluation Complete ---")
    print(f"Total samples processed: {num_evaluated + skipped_samples}")
    print(f"Samples skipped (e.g., no EOS found, empty prompt): {skipped_samples}")
    print(f"Samples evaluated: {num_evaluated}")

    metrics_to_log = {}
    if num_evaluated > 0 and len(all_gt_pads) > 0:
        # Calculate Precision, Recall, F1 for the PAD class in the padding zone
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_gt_pads, all_pred_pads, average='binary', pos_label=1, zero_division=0
        )
        # Calculate overall accuracy in the padding zone
        accuracy = accuracy_score(all_gt_pads, all_pred_pads)

        # Calculate length metrics
        mean_len_diff = np.mean(all_length_diffs)
        abs_mean_len_diff = np.mean(np.abs(all_length_diffs))

        metrics_to_log["eval_pad/precision"] = precision
        metrics_to_log["eval_pad/recall"] = recall
        metrics_to_log["eval_pad/f1_score"] = f1
        metrics_to_log["eval_pad/accuracy"] = accuracy
        metrics_to_log["eval_pad/mean_length_diff"] = mean_len_diff
        metrics_to_log["eval_pad/abs_mean_length_diff"] = abs_mean_len_diff

        print(f"\n--- Final Padding Prediction Metrics (Padding Zone Only) ---")
        print(f"PAD Precision: {precision:.4f}")
        print(f"PAD Recall:    {recall:.4f}")
        print(f"PAD F1-Score:  {f1:.4f}")
        print(f"PAD Accuracy:  {accuracy:.4f}")
        print(f"\n--- Sequence Length Metrics ---")
        print(f"Mean Length Difference (Pred - GT): {mean_len_diff:.2f} tokens")
        print(f"Mean Absolute Length Difference:    {abs_mean_len_diff:.2f} tokens")


        if log_to_wandb:
            wandb.log(metrics_to_log)
            print("\nMetrics logged to WandB.")

    else:
        print("\nNo valid samples were evaluated or no padding zone data. Cannot calculate metrics.")

    if log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Final Padding Generation of a Diffusion LLM.")
    parser.add_argument("eval_config_path", type=str, help="Path to the .json config file containing evaluation parameters.")
    args = parser.parse_args()

    eval_config_path_abs = os.path.abspath(args.eval_config_path)
    if not os.path.exists(eval_config_path_abs):
        print(f"[!] Error: Evaluation config file not found at {eval_config_path_abs}")
        sys.exit(1)
    try:
        with open(eval_config_path_abs, "r") as f:
            eval_config = json.load(f)
            eval_config["_config_path_origin"] = eval_config_path_abs
    except Exception as e:
        print(f"[!] Error loading or parsing config file {eval_config_path_abs}: {e}")
        sys.exit(1)

    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    evaluate_final_padding(eval_config, device)