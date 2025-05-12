#!/usr/bin/env python
"""
Script to precompute and store LLaDA embeddings from train, validation, and test datasets
for more efficient training of classification and regression heads later.
"""

import os
import argparse
import torch
import h5py
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Import your dataset module
# Change this import to match your actual project structure
from diffusion_llms.dataloader.llada_2 import DataModule
from diffusion_llms.input_helper import get_config


def extract_embeddings(model, dataloader, device, batch_size, context_length):
    """
    Extract and return embeddings from the LLaDA model for a given dataset.
    Returns embeddings and labels.
    """
    all_last_hidden = []
    all_pooled = []
    all_eos_labels = []
    all_true_lengths = []
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        eos_labels = batch["eos_labels"]
        true_length = batch["true_length"]
        
        # Forward pass - with no grad for efficiency
        with torch.no_grad():
            # Enable output_hidden_states to get all hidden states
            outputs = model(
                input_ids=input_ids,
                return_dict=True,
                output_hidden_states=True
            )
            
            # Get the last hidden state (last layer's output)
            last_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            
            # Create a mean-pooled representation
            pooled = torch.mean(last_hidden, dim=1)  # [batch_size, hidden_dim]
        
        # Store results

        all_last_hidden.append(last_hidden.cpu())
        all_pooled.append(pooled.cpu())
        all_eos_labels.append(eos_labels)
        all_true_lengths.append(true_length)
    
    # Concatenate results
    return {
        "last_hidden": all_last_hidden,
        "pooled": all_pooled,
        "eos_labels": all_eos_labels,
        "true_lengths": all_true_lengths
    }


def save_to_h5(data, output_file):
    """Save embeddings and labels to an HDF5 file"""
    with h5py.File(output_file, 'w') as f:
        # Create groups for organization
        embeddings_group = f.create_group('embeddings')
        labels_group = f.create_group('labels')
        
        # Save each batch of embeddings as separate datasets to avoid memory issues
        for i, (last_hidden, pooled, eos_labels, true_lengths) in enumerate(zip(
            data["last_hidden"], data["pooled"], data["eos_labels"], data["true_lengths"]
        )):
            # Create batch group
            batch_group = embeddings_group.create_group(f'batch_{i}')
            
            # Save embeddings
            batch_group.create_dataset('last_hidden', data=last_hidden.numpy(), compression='gzip')
            
            # Save labels
            labels_batch_group = labels_group.create_group(f'batch_{i}')
            labels_batch_group.create_dataset('eos_labels', data=eos_labels.numpy(), compression='gzip')
            labels_batch_group.create_dataset('true_lengths', data=true_lengths.numpy(), compression='gzip')


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Prepare embedded dataset for LLaDA training')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='embedded_datasets', help='Directory to save embedded datasets')
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct', help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = get_config(args.config)
    
    # Update batch size if provided
    if args.batch_size:
        config["batch_size"] = args.batch_size
    
    # Use GPU if available
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()  # Set to evaluation mode
    model.to(device)
    
    # Load data
    print("Initializing data module")
    datamodule = DataModule(config, tokenizer)
    datamodule.setup()
    
    # Process each split
    splits = {
        'train': datamodule.train_dataloader(),
        'val': datamodule.val_dataloader(),
        'test': datamodule.test_dataloader()
    }
    
    # Process and save each dataset
    for split_name, dataloader in splits.items():
        print(f"Processing {split_name} dataset")
        
        # Extract embeddings
        embeddings_data = extract_embeddings(
            model, 
            dataloader, 
            device, 
            config["batch_size"],
            config["context_length"]
        )
        
        # Save to file
        output_file = os.path.join(args.output_dir, f'{split_name}_embeddings.h5')
        print(f"Saving {split_name} embeddings to {output_file}")
        save_to_h5(embeddings_data, output_file)
        
        print(f"Completed {split_name} dataset")
    
    print("All embeddings have been computed and saved!")
    print(f"Files are saved in: {args.output_dir}")


if __name__ == "__main__":
    main()