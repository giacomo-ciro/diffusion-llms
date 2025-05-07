import sys
import os
import argparse
import torch

try:
    # Assuming your file is llada_datamodule.py in the dataloader directory
    from llada_datamodule import DataModule
    import tiktoken # Need tokenizer for decoding example
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure test_llada_datamodule.py is run from the correct directory relative to llada_datamodule.py")
    sys.exit(1)

def test_datamodule(config_path: str, num_batches_to_check: int = 2):
    """
    Tests the DataModule by loading data and checking batch shapes.
    """
    print(f"--- Testing Llada DataModule ---")
    print(f"Using config: {config_path}")

    try:
        # 1. Instantiate the DataModule
        dm = DataModule(config_path=config_path)
        print("DataModule instantiated successfully.")

        # 2. Call setup()
        dm.setup()
        print("DataModule setup complete.")
        print(f"Task type from config: {dm.config.get('task_type')}")

        # 3. Get the training dataloader
        train_loader = dm.train_dataloader()
        print(f"Train dataloader created. Number of batches: {len(train_loader)}")

        if len(train_loader) == 0:
            print("ERROR: Training dataloader is empty!")
            return

        # 4. Iterate through a few batches
        print(f"\n--- Checking first {num_batches_to_check} batches ---")
        for i, batch in enumerate(train_loader):
            if i >= num_batches_to_check:
                break

            print(f"\nBatch {i+1}:")
            # Batch should contain X, y, msk
            if len(batch) != 3:
                print(f"ERROR: Batch should have 3 elements (X, y, msk), but got {len(batch)}")
                continue

            X, y, msk = batch

            # 5. Check shapes and types
            print(f"  X shape: {X.shape}, dtype: {X.dtype}")
            print(f"  y shape: {y.shape}, dtype: {y.dtype}")
            print(f"  msk shape: {msk.shape}, dtype: {msk.dtype}")

            # --- Verification based on task type ---
            task_type = dm.config.get('task_type')
            expected_context_length = dm.config.get('context_length', 1024)
            expected_batch_size = X.shape[0] # Get actual batch size

            # Check X
            assert X.dim() == 2 and X.shape[1] == expected_context_length, f"X shape mismatch! Expected [~{expected_batch_size}, {expected_context_length}]"
            assert X.dtype == torch.long, "X dtype mismatch! Expected torch.long"

            # Check y shape based on task
            if task_type == "regression":
                assert y.dim() == 2 and y.shape[1] == 1, f"Regression y shape mismatch! Expected [~{expected_batch_size}, 1]"
                assert y.dtype == torch.float, "Regression y dtype mismatch! Expected torch.float"
            elif task_type == "classification":
                assert y.dim() == 2 and y.shape[1] == expected_context_length, f"Classification y shape mismatch! Expected [~{expected_batch_size}, {expected_context_length}]"
                assert y.dtype == torch.long, "Classification y dtype mismatch! Expected torch.long"
            else:
                 print(f"WARNING: Unknown task type '{task_type}' for detailed y check.")

            # Check msk
            assert msk.dim() == 2 and msk.shape[1] == expected_context_length, f"msk shape mismatch! Expected [~{expected_batch_size}, {expected_context_length}]"
            assert msk.dtype == torch.bool, "msk dtype mismatch! Expected torch.bool"
            assert msk.shape == X.shape, "msk shape should match X shape"

            # 6. (Optional) Decode and print one sample from the batch
            print("\n  --- Example Sample (Index 0 in Batch) ---")
            try:
                tokenizer = dm.tokenizer # Get tokenizer from datamodule
                X_sample = X[0].tolist()
                # Filter out padding for cleaner printing
                X_sample_unpadded = [tok for tok in X_sample if tok != dm.config.get("pad_token_id", 50257)]
                print(f"  Decoded X (unpadded): '{tokenizer.decode(X_sample_unpadded)}'")
                # Print target y differently based on task
                if task_type == "regression":
                    print(f"  Target y (length): {y[0].item()}")
                elif task_type == "classification":
                    # For classification, y[0] is the label sequence. Let's print where it's 1 (EOS).
                    y_sample_labels = y[0].tolist() # Convert tensor to list
                    eos_indices_in_y = [i for i, label in enumerate(y_sample_labels) if label == 1]
                    print(f"  Target y (EOS positions): {eos_indices_in_y}") # Print indices where label is 1
                else:
                    print(f"  Target y: {y[0]}") # Fallback for unknown task
                # Find where mask is True
                msk_indices = torch.where(msk[0])[0].tolist()
                if msk_indices:
                    print(f"  Masked indices (answer part): {min(msk_indices)} to {max(msk_indices)}")
                else:
                    print("  Masked indices (answer part): None")

                # If classification, show where y is 1 within the masked part
                if task_type == "classification":
                     y_sample = y[0]
                     eos_pos = torch.where(y_sample[msk[0]])[0] # Find 1s within the masked area
                     if eos_pos.numel() > 0:
                          # Add start index of mask to get absolute position
                          abs_eos_pos = eos_pos + min(msk_indices)
                          print(f"  EOS position(s) in y (where y=1 within mask): {abs_eos_pos.tolist()}")
                     else:
                          print(f"  EOS position(s) in y (where y=1 within mask): None")

            except Exception as e:
                print(f"  Error during sample decoding/printing: {e}")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Llada DataModule.")
    parser.add_argument("config_path", type=str, help="Path to the configuration JSON file (e.g., configs/test_llada_config.json).")
    args = parser.parse_args()
    test_datamodule(args.config_path)