import os
import yaml
import argparse

def parse_args():
    """Parse command-line arguments for configuration handling."""
    parser = argparse.ArgumentParser(description='Train VarLenLLada model with config file')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (overrides config file)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides config file)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            print(f"Configuration loaded from {config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

def override_config_with_args(config, args):
    """Override configuration with command-line arguments."""
    if args.resume:
        config['checkpoint']['resume_from'] = args.resume
        print(f"Overriding resume checkpoint: {args.resume}")
    
    if args.checkpoint_dir:
        config['checkpoint']['dir'] = args.checkpoint_dir
        print(f"Overriding checkpoint directory: {args.checkpoint_dir}")
    
    return config

def flatten_config(config):
    """Convert nested config to flat dict for compatibility with existing code."""
    flat_config = {}
    
    # Model config
    flat_config["pretrained_model_name"] = config["model"]["pretrained_model_name"]
    flat_config["context_length"] = config["model"]["context_length"]
    flat_config["context_length"] = config["model"]["context_length"]
    
    # Training config
    flat_config["batch_size"] = config["training"]["batch_size"]
    flat_config["lr"] = config["training"]["learning_rate"]
    flat_config["n_epochs"] = config["training"]["epochs"]
    flat_config["accumulate_grad"] = config["training"]["accumulate_grad"]
    flat_config["n_steps"] = config["training"]["max_steps"]
    flat_config["seed"] = config["training"]["seed"]
    
    # Validation config
    flat_config["val_test_perc"] = config["validation"]["val_test_perc"]
    flat_config["val_check_steps"] = config["validation"]["val_check_steps"]
    
    # Checkpoint config
    flat_config["checkpoint_dir"] = config["checkpoint"]["dir"]
    flat_config["checkpoint_steps"] = config["checkpoint"]["steps"]
    flat_config["resume_from_checkpoint"] = config["checkpoint"]["resume_from"]
    
    # Early stopping config
    flat_config["early_stopping_patience"] = config["early_stopping"]["patience"]
    flat_config["early_stopping_min_delta"] = config["early_stopping"]["min_delta"]
    
    # Logging config
    flat_config["run_name"] = config["logging"]["run_name"]
    flat_config["project_name"] = config["logging"]["project_name"]
    flat_config["log_dir"] = config["logging"]["log_dir"]
    
    return flat_config

def get_config():
    """Main function to get configuration from file and command-line args."""
    #args = parse_args()
    config = load_config('llada_config.yml')
    #config = override_config_with_args(config, args)
    
    # Create necessary directories
    os.makedirs(config["checkpoint"]["dir"], exist_ok=True)
    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    
    # Return the flattened config for compatibility with existing code
    return flatten_config(config)