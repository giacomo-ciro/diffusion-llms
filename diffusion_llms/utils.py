import torch
import sys
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

def check_config_validity(
        config:dict
):  
    message = ""
    if ( bool(config["resume_training"]) and
    not bool(config["init_from"]) ):
        message += "\'init_from\' with path/to/model.ckpt required when resume_training == True" + "\n"
    
    if config["n_embd"] % config["n_head"] != 0:
        message += "\'n_heads\' must be a divisor of \'n_embd\'" + "\n"
    
    tmp = {"diffusion", "arm"}
    if config["pipeline"] not in tmp:
        message += f"\'pipeline\' must be in {tmp}" + "\n"
    
    if config["pipeline"] == "diffusion":
        if not config["attn_annealing_steps"] >= 0:
            message += "\'attn_annealing_steps\' > 0 required when \'pipeline\' == diffusion"
    
    # TODO:complete list
    types = {
        "int": ["context_length"],
        "float": ["max_lr"],
        "str": ["memmap_path"]
    }
    for expected_type_name, attributes in types.items():
        expected_type = eval(expected_type_name)
        for attribute in attributes:
            if not isinstance(config[attribute], expected_type):
                message += f"\'{attribute}\' must be {expected_type}"
    
    # TODO: check validity of all other arguments

    if message:
        print(f"[!] Error in config.json:\n{message}")
        sys.exit()
    
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