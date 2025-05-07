#!/usr/bin/env python
# filepath: counter.py
"""
Token Counter Script

This script counts the total number of tokens in a dataset specified in a config file.
Usage: python counter.py path/to/config.json
"""

import json
import sys
import os
import pandas as pd
import tiktoken


def count_tokens_in_dataset(config_path):
    """
    Count tokens in a dataset using parameters from config file
    
    Args:
        config_path (str): Path to the configuration JSON file
        
    Returns:
        int: Total number of tokens in the dataset
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Get dataset path
    csv_path = config["llada_train_path"]
    # Handle relative paths
    if not os.path.isabs(csv_path) and not csv_path.startswith('./'):
        base_dir = os.path.dirname(os.path.abspath(config_path))
        csv_path = os.path.join(base_dir, csv_path)
    elif csv_path.startswith('./'):
        base_dir = os.path.dirname(os.path.abspath(config_path))
        csv_path = os.path.join(base_dir, csv_path[2:])
    
    # Load CSV data
    try:
        data_df = pd.read_csv(csv_path)
        print(f"Successfully loaded dataset from {csv_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_columns = ["user_prompt", "model_response"]
    for col in required_columns:
        if col not in data_df.columns:
            print(f"Error: Required column '{col}' not found in dataset")
            sys.exit(1)
    
    # Extract prompts and responses
    prompts = data_df["user_prompt"].astype(str).tolist()
    responses = data_df["model_response"].astype(str).tolist()
    
    # Initialize tokenizer (using GPT-2 encoding as default)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Count tokens
    prompt_tokens = 0
    response_tokens = 0
    eos_tokens = len(prompts)  # One EOS token per sample
    
    for prompt, response in zip(prompts, responses):
        prompt_tokens += len(tokenizer.encode(prompt))
        response_tokens += len(tokenizer.encode(response))
    
    # Calculate totals
    total_tokens = prompt_tokens + response_tokens + eos_tokens
    
    # Print results
    print("\n===== Token Count Results =====")
    print(f"Number of samples: {len(prompts)}")
    print(f"Prompt tokens: {prompt_tokens:,}")
    print(f"Response tokens: {response_tokens:,}")
    print(f"EOS tokens: {eos_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per sample: {total_tokens / len(prompts):.2f}")
    
    return total_tokens


def main():
    """Main function that parses arguments and runs token counter"""
    if len(sys.argv) != 2:
        print("Usage: python counter.py path/to/config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    total_count = count_tokens_in_dataset(config_path)
    print("\nToken counting completed successfully.")


if __name__ == "__main__":
    main()