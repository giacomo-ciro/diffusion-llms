import torch
import sys
import os
from typing import Any

def compute_binary_metrics(flat_preds, flat_targets):
    """
    Given 1d predictions and targets returns accuracy, recall, precision, f1.
    """
    # Calculate base metrics
    correct_preds = torch.eq(flat_preds, flat_targets) 
    tp = torch.sum(
        correct_preds[flat_targets]
    )
    tn = torch.sum(
        correct_preds[~flat_targets]
    )
    fp = torch.sum(
        ~correct_preds[~flat_targets]
    )
    fn = torch.sum(
        ~correct_preds[flat_targets]
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)

    return accuracy, recall, precision, f1

def get_annealing_mask(
        context_length:int,
        B:int,
        p:int
    )-> torch.Tensor:
    """
    Returns the annealed mask of shape [B, 1, context_length, context_length] 
    to be broadcasted to the attention heads.
        - context_length (int): the context length
        - B (int): the batch size
        - p (int): the probability of unmasking an entry in the attention 
                   mask (p=1.0 allows the model to see everything) in the 
                   annealed mask will be unmasked.
    """
    attn_mask = torch.tril(torch.ones(context_length, context_length)).to(torch.bool)        
    random_mask = torch.rand(size=(context_length, context_length)) <= p
    anneal_mask = torch.logical_or(attn_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(B, 1, context_length, context_length)
    return expanded_mask

def get_causal_mask(
        context_length:int,
        B:int
    )-> torch.Tensor:
    """
    Return the casual mask (lower triangular) of shape [B, 1, context_length, context_length] 
    to be broadcasted to the attention heads.
        - context_length (int): the context length
        - B (int): the batch size
    """
    attn_mask = torch.tril(torch.ones(context_length, context_length)).to(torch.bool)        
    expanded_mask = attn_mask[None, None, :, :].expand(B, 1, context_length, context_length)
    return expanded_mask

def check_config_validity(config:dict):  
    """
    Performs comprehensive validation of configuration parameters used throughout the project.
    
    This function checks:
    1. Required configuration keys
    2. Parameter type correctness
    3. Value validity (ranges, formats)
    4. Inter-parameter dependencies
    5. File path existence (where applicable)
    6. Model-specific parameter validation
    
    Args:
        config (dict): Configuration dictionary loaded from JSON
        
    Returns:
        None: Exits with sys.exit() if validation fails, prints success message otherwise
    """
    message = ""
    
    # --- Core Model Configuration ---
    # Check if required keys exist
    required_keys = [
        "pipeline", "init_from", "context_length", "n_embd", "n_layer", "n_head", 
        "eos_token_id", "mask_id"
    ]
    for key in required_keys:
        if key not in config:
            message += f"Required key '{key}' is missing in config" + "\n"
    
    # Check pipeline type
    valid_pipelines = {"diffusion", "arm", "diffusion_length_head"}
    if "pipeline" in config and config["pipeline"] not in valid_pipelines:
        message += f"'pipeline' must be one of {valid_pipelines}" + "\n"
    
    # Check embedding dimension related to heads
    if "n_embd" in config and "n_head" in config and config["n_embd"] % config["n_head"] != 0:
        message += "'n_head' must be a divisor of 'n_embd'" + "\n"
    
    # Check init_from with resume_training
    if "resume_training" in config and "init_from" in config:
        if bool(config["resume_training"]) and not bool(config["init_from"]):
            message += "'init_from' with path/to/model.ckpt required when resume_training == True" + "\n"
    
    # Check diffusion specific parameters
    if "pipeline" in config and config["pipeline"] == "diffusion":
        if "attn_annealing_steps" in config and not config["attn_annealing_steps"] >= 0:
            message += "'attn_annealing_steps' must be >= 0 when 'pipeline' == diffusion" + "\n"
        
        if "denoising_strategy" in config and config["denoising_strategy"] not in ["random", "entropy"]:
            message += "'denoising_strategy' must be either 'random' or 'entropy'" + "\n"
            
        if "diffusion_steps" in config and config["diffusion_steps"] < 1:
            message += "'diffusion_steps' must be at least 1 for diffusion pipeline" + "\n"
    
    # Check padded_dataset related parameters
    if "padded_dataset" in config and config["padded_dataset"]:
        if "pad_token_id" not in config:
            message += "'pad_token_id' required when padded_dataset == True" + "\n"
        if "pad_masked_perc" in config and (config["pad_masked_perc"] < 0 or config["pad_masked_perc"] > 1):
            message += "'pad_masked_perc' must be between 0 and 1" + "\n"
    
    # Check dataset path existence if specified
    if "memmap_path" in config:
        if not os.path.exists(config["memmap_path"]):
            message += f"Dataset path '{config['memmap_path']}' does not exist" + "\n"
    
    # --- Training Parameters ---
    # Check optimization parameters
    if "max_lr" in config and config["max_lr"] <= 0:
        message += "'max_lr' must be positive" + "\n"
    
    if "batch_size" in config and config["batch_size"] < 1:
        message += "'batch_size' must be at least 1" + "\n"
    
    # Check either n_epochs or n_steps is specified and valid
    if "n_steps" in config and "n_epochs" in config:
        if config["n_steps"] < 1 and config["n_epochs"] < 1:
            message += "Either 'n_steps' or 'n_epochs' must be at least 1" + "\n"
    
    if "accumulate_grad" in config and config["accumulate_grad"] < 1:
        message += "'accumulate_grad' must be at least 1" + "\n"
    
    if "grad_clip" in config and config["grad_clip"] < 0:
        message += "'grad_clip' must be non-negative" + "\n"
    
    if "val_check_interval" in config and config["val_check_interval"] < 1:
        message += "'val_check_interval' must be at least 1" + "\n"
    
    if "warmup_pct" in config and (config["warmup_pct"] < 0 or config["warmup_pct"] > 1):
        message += "'warmup_pct' must be between 0 and 1" + "\n"
    
    # Check betas for Adam optimizer
    if "betas" in config:
        if not isinstance(config["betas"], list) or len(config["betas"]) != 2:
            message += "'betas' must be a list with exactly 2 values" + "\n"
        elif any(b < 0 or b >= 1 for b in config["betas"]):
            message += "Each value in 'betas' must be between 0 and 1 (exclusive)" + "\n"
    
    # --- Generation Parameters ---
    if "temperature" in config and config["temperature"] <= 0:
        message += "'temperature' must be positive" + "\n"
    
    if "max_new_tokens" in config and config["max_new_tokens"] < 1:
        message += "'max_new_tokens' must be at least 1" + "\n"
    
    if "repetition_penalty" in config and config["repetition_penalty"] < 0:
        message += "'repetition_penalty' must be non-negative" + "\n"
    
    # --- Special Token IDs ---
    # Check for valid token IDs:
    # - mask_id should be a valid GPT-2 token ID (typically < 50257)
    # - eos_token_id should be the standard GPT-2 EOS token (50256)
    # - pad_token_id should be 50257 (added token)
    if "mask_id" in config and (config["mask_id"] < 0 or config["mask_id"] >= 50257):
        message += "'mask_id' must be a valid token ID (0-50256)" + "\n"
    
    if "eos_token_id" in config and config["eos_token_id"] != 50256:
        message += "Warning: 'eos_token_id' is not the standard GPT-2 EOS token (50256)" + "\n"
    
    if "pad_token_id" in config and config["pad_token_id"] != 50257:
        message += "Warning: 'pad_token_id' is not the standard added token position (50257)" + "\n"
    
    # --- LLaDa-specific parameters ---
    if "eos_window_max" in config and config["eos_window_max"] < 1:
        message += "'eos_window_max' must be at least 1" + "\n"
    
    if "window_annealing_steps" in config and config["window_annealing_steps"] < 0:
        message += "'window_annealing_steps' must be non-negative" + "\n"
    
    if "random_mask_prob" in config and (config["random_mask_prob"] < 0 or config["random_mask_prob"] > 1):
        message += "'random_mask_prob' must be between 0 and 1" + "\n"
    
    # --- Evaluation-specific parameters ---
    if "eval_batch_size" in config and config["eval_batch_size"] < 1:
        message += "'eval_batch_size' must be at least 1" + "\n"
    
    if "ckpt_path_for_eval" in config and "UPDATE_THIS" in config["ckpt_path_for_eval"]:
        message += "'ckpt_path_for_eval' needs to be updated with an actual path" + "\n"
    
    # --- File path validations ---
    # For checkpoint saving
    if "save_dir" in config and "enable_checkpointing" in config and config["enable_checkpointing"]:
        save_dir = config["save_dir"]
        if not os.path.exists(save_dir) and not os.path.isdir(save_dir):
            try:
                # Try to create the directory
                os.makedirs(save_dir, exist_ok=True)
            except:
                message += f"Cannot create save_dir '{save_dir}' for checkpointing" + "\n"
    
    # --- Type checking ---
    types = {
        "int": [
            "context_length", "n_embd", "n_layer", "n_head", "attn_annealing_steps", 
            "mask_id", "eos_token_id", "pad_token_id", "n_epochs", "n_steps", 
            "val_check_interval", "batch_size", "accumulate_grad", "diffusion_steps",
            "max_new_tokens", "n_samples", "eos_window_max", "window_annealing_steps"
        ],
        "float": [
            "max_lr", "warmup_pct", "weight_decay", "div_factor", "final_div_factor",
            "temperature", "repetition_penalty", "pad_masked_perc", "val_test_perc",
            "random_mask_prob", "grad_clip"
        ],
        "str": [
            "memmap_path", "pipeline", "init_from", "save_dir", "run_name", 
            "project_name", "user_prompt", "denoising_strategy"
        ],
        "bool": [
            "resume_training", "enable_checkpointing", "wandb", "padded_dataset", 
            "do_sample", "use_pad_head"
        ]
    }
    
    for expected_type_name, attributes in types.items():
        expected_type = eval(expected_type_name)
        for attribute in attributes:
            if attribute in config and not isinstance(config[attribute], expected_type):
                message += f"'{attribute}' must be {expected_type_name}" + "\n"
    
    if message:
        print(f"[!] Error in config.json:\n{message}")
        #sys.exit()
    print("The provided configuration file is valid!")

def crop_to_var_len(model: Any, eos_token_id: int, x: torch.Tensor, attention_mask: torch.Tensor, pad_token_id: int = None) -> torch.Tensor:
    """
    Infers variable-length sequences by determining the position of the EOS token
    for each sentence in the batch, truncating or padding as necessary.

    Args:
        model (Any): The model to infer logits.
        eos_token_id (int): The ID of the EOS token.
        x (torch.Tensor): Input tensor of shape [B, T] (batch size, sequence length).
        pad_token_id (int, optional): The ID of the padding token. If None, no padding is applied.

    Returns:
        torch.Tensor: Updated tensor with truncated or padded sequences.
    """
    B, T = x.shape
    with torch.no_grad():
        logits, _ = model(x, attention_mask=attention_mask)  # Infer logits from the model with attention mask

    # Find EOS positions for each sentence
    eos_positions = torch.argmax(logits[:, :, eos_token_id], dim=-1)  # (B,)

    # Create a mask to truncate or pad sequences
    max_length = eos_positions.max().item() + 1  # Longest sequence length including EOS
    x = x[:, :max_length] # new truncated x

    # Create a mask of True values to the right of the EOS token position
    left_eos_mask = torch.arange(T).expand(B, T) > eos_positions[:, None]

    # Replace values to the right of EOS with the EOS token ID 
    x[left_eos_mask] = pad_token_id or eos_token_id

    return x # [B, max_length]

def get_device():
    """Get the device to use for PyTorch operations (CPU or cuda or MPS)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device