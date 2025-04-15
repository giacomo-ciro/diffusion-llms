import torch
import sys

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

def get_causal_mask(\
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