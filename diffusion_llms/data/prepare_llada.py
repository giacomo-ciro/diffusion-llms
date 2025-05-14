#!/usr/bin/env python
"""
Script to precompute and store LLaDA embeddings from train, validation, and test datasets
for more efficient training of classification and regression heads later.
Saves embeddings in separate files every N batches, supports resume and reproducible seeds.
"""

import os
import argparse
import torch
import h5py
import json
import glob
import re
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Import your dataset module
from diffusion_llms.dataloader.llada_dataloader import DataModule
from diffusion_llms.input_helper import get_config


def extract_embeddings(
    model, dataloader, device,
    output_dir, split_name,
    save_every=50,
    resume=False,
    max_batches=None
):
    """
    Extract embeddings and save in separate HDF5 files every `save_every` batches.
    Supports resume by skipping already-created file parts.
    Stops early if `max_batches` is reached.
    """
    # Determine which parts already exist
    resume_parts = set()
    if resume:
        pattern = os.path.join(output_dir, f"{split_name}_embeddings_part*.h5")
        for path in glob.glob(pattern):
            m = re.search(rf"{split_name}_embeddings_part(\d+)\.h5$", path)
            if m:
                resume_parts.add(int(m.group(1)))
    
    total_batches = len(dataloader)
    file_idx = -1
    h5f = None
    embeddings_group = None
    labels_group = None

    for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"Processing {split_name}")):
        # Stop if reached max_batches
        if max_batches is not None and batch_idx >= max_batches:
            break

        part_idx = batch_idx // save_every
        # Skip entire part if resuming
        if resume and part_idx in resume_parts:
            continue

        # On new part, open new file
        if part_idx != file_idx:
            if h5f:
                h5f.close()
            chunk_name = f"{split_name}_embeddings_part{part_idx}.h5"
            chunk_path = os.path.join(output_dir, chunk_name)
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
            h5f = h5py.File(chunk_path, 'w')
            embeddings_group = h5f.create_group('embeddings')
            labels_group = h5f.create_group('labels')
            file_idx = part_idx

        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        eos_labels = batch["eos_labels"]
        true_length = batch["true_length"]

        # Forward pass without gradients
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                return_dict=True,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

        # Save embeddings
        grp = embeddings_group.create_group(f'batch_{batch_idx}')
        grp.create_dataset('last_hidden', data=last_hidden.cpu().numpy(), compression='gzip')

        # Save labels
        lbl_grp = labels_group.create_group(f'batch_{batch_idx}')
        lbl_grp.create_dataset('eos_labels', data=eos_labels.numpy(), compression='gzip')
        lbl_grp.create_dataset('true_lengths', data=true_length.numpy(), compression='gzip')

    # Close any open file
    if h5f:
        h5f.close()


def main():
    parser = argparse.ArgumentParser(description='Prepare embedded dataset for LLaDA training')
    parser.add_argument('--output_dir', type=str, default='embedded_datasets_new',
                        help='Directory to save embedded datasets')
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Number of batches per output file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing part files')
    parser.add_argument('--max_batches', type=int, default=100,
                        help='Stop after processing this many batches')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare
    os.makedirs(args.output_dir, exist_ok=True)
    config = get_config()
    config['batch_size'] = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()
    model.to(device)

    print("Initializing data module")
    datamodule = DataModule(config, tokenizer)
    datamodule.setup()

    splits = {
        'train': datamodule.train_dataloader(),
        'val': datamodule.val_dataloader(),
        'test': datamodule.test_dataloader()
    }

    for split_name, dataloader in splits.items():
        print(f"Processing split: {split_name}")
        extract_embeddings(
            model=model,
            dataloader=dataloader,
            device=device,
            output_dir=args.output_dir,
            split_name=split_name,
            save_every=args.save_every,
            resume=args.resume,
            max_batches=args.max_batches
        )
        print(f"Completed {split_name} embeddings.")

    print("All embeddings have been computed and saved!")