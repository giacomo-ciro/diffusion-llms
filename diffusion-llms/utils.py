import sys

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