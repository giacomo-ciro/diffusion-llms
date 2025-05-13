#!/usr/bin/env python
"""
Script to precompute and store LLaDA embeddings from train, validation, and test datasets
for more efficient training of classification and regression heads later.
"""

import os
import argparse
import torch
import h5py
import json
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Import your dataset module
# Change this import to match your actual project structure
from diffusion_llms.dataloader.llada_2 import DataModule
from diffusion_llms.input_helper import get_config


def extract_embeddings(model, dataloader, device, batch_size, context_length, 
                      output_file, checkpoint_every=100, resume_from=None):
    """
    Extract and return embeddings from the LLaDA model for a given dataset.
    Saves checkpoints every 'checkpoint_every' steps to allow resuming.
    """
    all_last_hidden = []
    all_pooled = []
    all_eos_labels = []
    all_true_lengths = []
    
    # If resuming, load checkpoint info
    start_batch = 0
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, 'r') as f:
            checkpoint_info = json.load(f)
        start_batch = checkpoint_info.get('completed_batches', 0)
        print(f"Resuming from batch {start_batch}")
    
    # Create/open the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mode = 'a' if resume_from and os.path.exists(output_file) else 'w'
    
    with h5py.File(output_file, mode) as h5f:
        # Initialize groups if creating new file
        if mode == 'w':
            embeddings_group = h5f.create_group('embeddings')
            labels_group = h5f.create_group('labels')
        else:
            embeddings_group = h5f['embeddings']
            labels_group = h5f['labels']
        
        # Process batches
        total_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=total_batches, 
                   desc="Processing batches", initial=start_batch)
        
        # Skip already processed batches if resuming
        for batch_idx, batch in pbar:
            if batch_idx < start_batch:
                continue
                
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
                

            
            # Save directly to HDF5 to avoid memory issues
            batch_group = embeddings_group.create_group(f'batch_{batch_idx}')
            batch_group.create_dataset('last_hidden', data=last_hidden.cpu().numpy(), compression='gzip')
            
            labels_batch_group = labels_group.create_group(f'batch_{batch_idx}')
            labels_batch_group.create_dataset('eos_labels', data=eos_labels.numpy(), compression='gzip')
            labels_batch_group.create_dataset('true_lengths', data=true_length.numpy(), compression='gzip')
            
            # Save checkpoint every N batches
            if (batch_idx + 1) % checkpoint_every == 0 or batch_idx == total_batches - 1:
                # Create checkpoint file
                checkpoint_path = os.path.join(os.path.dirname(output_file), 
                                              f"{os.path.basename(output_file)}.checkpoint.json")
                checkpoint_info = {
                    'completed_batches': batch_idx + 1,
                    'total_batches': total_batches,
                    'output_file': output_file
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_info, f)
                
                pbar.set_description(f"Saved checkpoint at batch {batch_idx+1}/{total_batches}")
    
    # Return the checkpoint info path for potential future use
    return os.path.join(os.path.dirname(output_file), f"{os.path.basename(output_file)}.checkpoint.json")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Prepare embedded dataset for LLaDA training')
    parser.add_argument('--output_dir', type=str, default='embedded_datasets', help='Directory to save embedded datasets')
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct', help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint_every', type=int, default=100, help='Save checkpoint every N batches')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = get_config()
    
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
        #'train': datamodule.train_dataloader(),
        'val': datamodule.val_dataloader(),
        'test': datamodule.test_dataloader()
    }
    
    # Process and save each dataset
    for split_name, dataloader in splits.items():
        output_file = os.path.join(args.output_dir, f'{split_name}_embeddings.h5')
        checkpoint_file = os.path.join(args.output_dir, f"{split_name}_embeddings.h5.checkpoint.json")
        
        # Check if we can resume
        resume_from = None
        if args.resume and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_info = json.load(f)
            
            # Only resume if the file is incomplete
            if checkpoint_info.get('completed_batches', 0) < checkpoint_info.get('total_batches', 0):
                resume_from = checkpoint_file
                print(f"Found checkpoint for {split_name}, resuming...")
            else:
                print(f"Found completed checkpoint for {split_name}, skipping...")
                continue
        elif os.path.exists(output_file) and not args.resume:
            print(f"Output file for {split_name} already exists. Use --resume to continue from checkpoint.")
            continue
            
        print(f"Processing {split_name} dataset")
        
        # Extract embeddings with checkpointing
        extract_embeddings(
            model=model, 
            dataloader=dataloader, 
            device=device, 
            batch_size=config["batch_size"],
            context_length=config["context_length"],
            output_file=output_file,
            checkpoint_every=args.checkpoint_every,
            resume_from=resume_from
        )
        
        print(f"Completed {split_name} dataset")
    
    print("All embeddings have been computed and saved!")
    print(f"Files are saved in: {args.output_dir}")


if __name__ == "__main__":
    main()