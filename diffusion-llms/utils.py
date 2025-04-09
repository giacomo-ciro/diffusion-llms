import sys

def check_config_validity(
        config:dict
):  
    message = ""
    if ( bool(config["resume_training"]) and
    not bool(config["init_from"]) ):
        message += "Path/to/.ckpt required when resume_training == True" + "\n"
    
    if config["n_embd"] % config["n_head"] != 0:
        message += "n_heads must be a divisor of n_embd" + "\n"
    
    tmp = {"diffusion", "arm"}
    if config["pipeline"] not in tmp:
        message += "pipeline must be in {tmp}" + "\n"
    
    if config["pipeline"] == "diffusion":
        if not config["attn_annealing_steps"] >= 0:
            message += "attn_annealing_steps > 0 required when pipeline == diffusion"
    
    
    # TODO: check validity of all other arguments

    if message:
        print(f"[!] Error in config.json:\n{message}")
        sys.exit()
    
    return