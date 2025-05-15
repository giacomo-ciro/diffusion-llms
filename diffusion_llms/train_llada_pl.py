#!/usr/bin/env python
"""
Script to train a single LLaDA model (classifier, regressor, or full_regressor)
using PyTorch Lightning and precomputed embeddings.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
import hashlib

# Import the embedded data module
from diffusion_llms.input_helper import get_config
from diffusion_llms.dataloader.llada_dataloader import DataModule
import numpy as np
torch.set_float32_matmul_precision("medium")


class LladaBackbone(pl.LightningModule):
    def __init__(self, cache_dir="cache"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
        base_model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        # The transformer model
        self.transformer = base_model.model.transformer

        # The head
        self.lm_head = self.transformer.pop("ff_out")

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Freeze all parameters
        for param in self.transformer.parameters():
            param.requires_grad = False

    def _get_cache_path(self, input_ids: torch.Tensor) -> str:
        # Hash input ids
        input_hash = hashlib.md5(input_ids.cpu().numpy().tobytes()).hexdigest()
        return os.path.join(self.cache_dir, f"{input_hash}.pt")

    def _load_or_compute_hidden(self, input_ids: torch.Tensor):
        path = self._get_cache_path(input_ids)

        if os.path.exists(path):
            hidden = torch.load(path, map_location=self.device)
        else:
                hidden = self.forward_hidden_repr(input_ids)
                torch.save(hidden.cpu(), path)

        return hidden

    def forward(self, input_ids, target=None):
        return self._load_or_compute_hidden(input_ids)   

    def forward_hidden_repr(self, input_ids, attention_mask=None):
        """
        Forward pass through the transformer encoder to obtain the hidden states,
        excluding the final language modeling head (ff_out).
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) or a tuple
                    from which a tensor can be extracted.
            attention_mask: Optional tensor of shape (batch_size, sequence_length)
            
        Returns:
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_dim)
        """

        # Attempt to robustly extract the core tensor from input_ids if it's a tuple
        current_val = input_ids
        while isinstance(current_val, tuple):
            if not current_val: # Check for empty tuple
                raise ValueError("Cannot process an empty tuple as input_ids.")
            # Assume the primary data is the first element; continue unwrapping if it's also a tuple.
            current_val = current_val[0]
        
        if not torch.is_tensor(current_val):
            raise TypeError(
                f"Expected input_ids to resolve to a tensor, but got {type(input_ids)} "
                f"which resolved to {type(current_val)} ({current_val})."
            )
        
        actual_input_tensor = current_val

        # Get embedding from wte (word token embedding)
        embedding_layer = self.transformer["wte"]
        hidden_states = embedding_layer(actual_input_tensor)
        
        # Apply dropout and layer norm if present
        if "emb_drop" in self.transformer:
            hidden_states = self.transformer["emb_drop"](hidden_states)
        # Based on your model's structure printout, ln_f is applied after embedding/dropout
        if "ln_f" in self.transformer:
            hidden_states = self.transformer["ln_f"](hidden_states)

        # Pass through the transformer blocks (encoder layers)
        for i, block in enumerate(self.transformer["blocks"]):
            input_to_block_type = type(hidden_states) # For debugging
            input_to_block_shape = hidden_states.shape if torch.is_tensor(hidden_states) else "N/A" # For debugging

            result = block(hidden_states)

            if isinstance(result, tuple):
                if not result: # Check for an empty tuple
                    raise ValueError(f"Transformer block {i} returned an empty tuple.")
                
                hidden_states = result[0] # Extract the first element

                # Rigorous check for the extracted hidden_states
                if not torch.is_tensor(hidden_states):
                    raise TypeError(
                        f"After processing block {i}, the first element of the returned tuple "
                        f"(expected to be hidden_states) is type {type(hidden_states)}, not a Tensor. "
                        f"The full tuple was: {result}. Input to block was type {input_to_block_type} with shape {input_to_block_shape}."
                    )
            else: # If block did not return a tuple
                hidden_states = result
                if not torch.is_tensor(hidden_states):
                    raise TypeError(
                        f"Transformer block {i} did not return a tuple, and its direct output "
                        f"is type {type(hidden_states)}, not a Tensor. "
                        f"Input to block was type {input_to_block_type} with shape {input_to_block_shape}."
                    )
                    
            # Optional: Print shape to verify
            # print(f"Block {i} output hidden_states shape: {hidden_states.shape}, type: {type(hidden_states)}")

        # Do not apply ff_out (final projection layer), as it's handled by self.lm_head
        return hidden_states


class LLaDaClassifier(nn.Module):
    """Classification head for LLaDA. Two‑layer MLP that predicts if each token is EOS (per‑token binary classification)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_state):
        # hidden_state: [B, T, D]

        # Check if shape is correct
        assert hidden_state.dim() == 3, f"Expected [B,T,D] tensor, got {hidden_state.dim()}D tensor"
        assert hidden_state.size(2) == self.hidden_size, f"Expected hidden size of {self.hidden_size}, got {hidden_state.size(2)}"

        # reshape so the MLP sees one token per row
        B, T, D = hidden_state.shape
        hidden_flat = hidden_state.view(-1, D)          # [B*T, D]
        logits_flat = self.net(hidden_flat).squeeze(-1)  # [B*T]
        logits = logits_flat.view(B, T)                  # [B, T]
        return logits

class LLaDaRegressor(nn.Module):
    """Predict log-token-count from pooled encoder output."""
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, pooled):
        assert pooled.ndim == 2, f"Expected [B, D], got {pooled.shape}"
        return self.net(pooled).squeeze(-1)   # log-length
        
class LLaDaTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training a single LLaDA model.
    """
    def __init__(
        self,
        model_type,
        hidden_size,
        context_length,
        cache_dir="cache",
        learning_rate=1e-5,
        pos_weight=10.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the selected model
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.pos_weight = torch.tensor([pos_weight])
        self.backbone = LladaBackbone(cache_dir)
        
        if model_type == "classifier":
            self.model = LLaDaClassifier(hidden_size)
        elif model_type == "regressor":
            self.model = LLaDaRegressor(hidden_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # get the hidden states
        # batch in here has input_ids, eos_labels, true_length, dummy attn matrix

        llada_hidden_states = self.backbone(batch["input_ids"]).detach()
        pooled_hidden_states = llada_hidden_states.mean(dim=1)  # [B, D]


        # Extract the appropriate data based on model type
        if self.model_type == "classifier":
            x = llada_hidden_states
            y = batch["eos_labels"].float()
            # Use all token representations as-is
            logits = self.model(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, 
                pos_weight=self.pos_weight.to(y.device)
            )
            
            # Log metrics
            self.log('train/loss', loss, prog_bar=True)
            
            # Calculate accuracy
            with torch.no_grad():
                pred_labels = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred_labels == y).float().mean()
                self.log('train/acc', acc, prog_bar=True)
                
        elif self.model_type == "regressor":
            x = pooled_hidden_states
            # training step
            y = batch["true_length"].float()
            y_log = torch.log1p(y)          # log(1 + length) for stability

            pred_log = self.model(pooled_hidden_states)  # <- use the same tensor you called x
            loss      = F.mse_loss(pred_log, y_log)      # MSE on log-scale targets
            self.log("train/loss_log_MSE", loss, prog_bar=True)
                
            # Calculate RMSE in original scale
            with torch.no_grad():
                pred_len = torch.expm1(pred_log)          # inverse of log1p
                rmse     = torch.sqrt(F.mse_loss(pred_len, y))
                mae      = F.l1_loss(pred_len, y)

                # (optional) coefficient of determination, protects against nan when var=0
                ss_tot = torch.sum((y - y.mean())**2)
                r2     = 1.0 - torch.sum((pred_len - y)**2) / (ss_tot + 1e-8)

                self.log_dict(
                                {
                                    "train/RMSE_tokens": rmse,
                                    "train/MAE_tokens":  mae,
                                    "train/R2_tokens":   r2,
                                },
                                prog_bar=True,      # show in tqdm
                                on_step=True,       # every optimisation step
                                on_epoch=True,      # and the epoch average
                            )

                
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Extract the appropriate data based on model type
        llada_hidden_states = self.backbone(batch["input_ids"]).detach()
        pooled_hidden_states = llada_hidden_states.mean(dim=1)  # [B, D]

        if self.model_type == "classifier":
            x = llada_hidden_states
            y = batch["eos_labels"].float()
            
            logits = self.model(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, 
                pos_weight=self.pos_weight.to(y.device)
            )
            
            # Log metrics
            self.log('val/loss', loss, prog_bar=True)
            
            # Calculate additional metrics
            with torch.no_grad():
                pred_labels = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred_labels == y).float().mean()
                self.log('val/acc', acc, prog_bar=True)
                
                # Calculate AUC for flattened predictions
                flat_logits = logits.view(-1)
                flat_labels = y.view(-1)
                try:
                    # AUC can fail if there's only one class in the batch
                    auc = roc_auc_score(
                        flat_labels.cpu().numpy(), 
                        torch.sigmoid(flat_logits).cpu().numpy()
                    )
                    self.log('val/auc', auc, prog_bar=True)
                except:
                    pass
                
        elif self.model_type == "regressor":
            y = batch["true_length"].float()
            y_log = torch.log1p(y)

            pred_log = self.model(pooled_hidden_states)
            loss = F.mse_loss(pred_log, y_log)
            self.log("val/loss_log_MSE", loss, prog_bar=True)

            # ---- metrics on original scale ----
            with torch.no_grad():
                pred_len = torch.expm1(pred_log)
                rmse = torch.sqrt(F.mse_loss(pred_len, y))
                mae = F.l1_loss(pred_len, y)
                ss_tot = torch.sum((y - y.mean())**2)
                r2 = 1.0 - torch.sum((pred_len - y)**2) / (ss_tot + 1e-8)

                self.log_dict(
                    {
                        "val/RMSE_tokens": rmse,
                        "val/MAE_tokens":  mae,
                        "val/R2_tokens":   r2,
                    },
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )
                
        elif self.model_type == "full_regressor":
            x = llada_hidden_states
            y = batch["true_length"].float()
            
            # Normalize target to 0-1 range
            y_normalized = y / self.context_length
            
            preds = self.model(x)
            preds = torch.exp(preds)
            loss = F.mse_loss(preds, y_normalized)
            
            # Log metrics
            self.log('val/loss', loss, prog_bar=True)
            
            # Calculate RMSE in original scale
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(torch.exp(preds) * self.context_length, y))
                self.log('val/rmse', rmse, prog_bar=True)
        
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        # Same as validation step, but for classifier, save logits and ids for CSV
        llada_hidden_states = self.backbone(batch["input_ids"]).detach()
        pooled_hidden_states = llada_hidden_states.mean(dim=1)  # [B, D]

        if self.model_type == "classifier":
            x = llada_hidden_states
            y = batch["eos_labels"].float()
            logits = self.model(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, 
                pos_weight=self.pos_weight.to(y.device)
            )
            self.log('test/loss', loss, prog_bar=True)
            with torch.no_grad():
                pred_labels = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred_labels == y).float().mean()
                self.log('test/acc', acc, prog_bar=True)
                # Save logits, labels, and optionally input ids for CSV
                # Save as attribute for later use in test_epoch_end
                if not hasattr(self, "test_logits"):
                    self.test_logits = []
                self.test_logits.append(logits.detach().cpu().flatten())
                # Save input_ids for reference (optional)
            return {'test_loss': loss}
        elif self.model_type == "regressor":
            x = pooled_hidden_states
            y = batch["true_length"].float()
            y_log = torch.log1p(y)

            pred_log = self.model(x)
            loss = F.mse_loss(pred_log, y_log)
            self.log('test/loss_log_MSE', loss, prog_bar=True)

            # ---- metrics on original scale ----
            with torch.no_grad():
                pred_len = torch.expm1(pred_log)
                rmse = torch.sqrt(F.mse_loss(pred_len, y))
                mae = F.l1_loss(pred_len, y)
                ss_tot = torch.sum((y - y.mean())**2)
                r2 = 1.0 - torch.sum((pred_len - y)**2) / (ss_tot + 1e-8)

                self.log_dict(
                    {
                        "test/RMSE_tokens": rmse,
                        "test/MAE_tokens":  mae,
                        "test/R2_tokens":   r2,
                    },
                    prog_bar=True,
                )

                if not hasattr(self, "regression_preds"):
                    self.regression_preds = []
                # save predictions in token space
                self.regression_preds.append(pred_len.detach().cpu().flatten())

            return {'test_loss': loss}
        elif self.model_type == "full_regressor":
            x = llada_hidden_states
            y = batch["true_length"].float()
            y_normalized = y / self.context_length
            preds = self.model(x)
            preds = torch.exp(preds)
            loss = F.mse_loss(preds, y_normalized)
            self.log('test/loss', loss, prog_bar=True)
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(preds * self.context_length, y))
                self.log('test/rmse', rmse, prog_bar=True)

                if not hasattr(self, "regression_preds"):
                    self.regression_preds = []

                self.regression_preds.append(preds.detach().cpu().flatten() * self.context_length)

            return {'test_loss': loss}

    def on_test_epoch_end(self):
        # For classifier, save logits and labels to CSV
        if self.model_type == "classifier" and hasattr(self, "test_logits"):
            logits_np = torch.cat(self.test_logits, dim=0).numpy()
            np.save(os.path.join(self.logger.save_dir, "test_logits.npy"), logits_np)

        elif self.model_type == "regressor" and hasattr(self, "regression_preds"):
            preds_np = torch.cat(self.regression_preds, dim=0).numpy()
            np.save(os.path.join(self.logger.save_dir, "regression_preds_pooled.npy"), preds_np)
        
        elif self.model_type == "full_regressor" and hasattr(self, "regression_preds"):
            preds_np = torch.cat(self.regression_preds, dim=0).numpy()
            np.save(os.path.join(self.logger.save_dir, "regression_preds_full.npy"), preds_np)
    
    def configure_optimizers(self):
        """Configure optimizer with OneCycleLR scheduler"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate # will be reset by onecycle lr
        )
        
        # Create the OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=5,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }   

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # keep only keys under "model" (i.e. your classifier/regressor head)
        state_dict = checkpoint["state_dict"]
        filtered = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        checkpoint["state_dict"] = filtered


def main():
    args = get_config()    
    print("Configuration loaded successfully.")
    print("args:", args)
    # Create config for data module
    
    # Create data module
    data_module = DataModule(
        args, 
        tokenizer=AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct"),
        num_workers=args["num_workers"]
    )
    data_module.setup()
    
    # Create model
    model = LLaDaTrainer(
        model_type=args["model_type"],
        hidden_size=args["hidden_size"],
        context_length=args["context_length"],
        learning_rate=args["learning_rate"],
        pos_weight=args["pos_weight"],
        cache_dir=args["cache_dir"]
    )
    
    # Set up logging
    logger = WandbLogger(
        project=args["project_name"],
        name=args["run_name"] or f"llada-{args['model_type']}",
        save_dir=args["output_dir"]
    )
    
    # Create callbacks
    callbacks = [
        # Model checkpoint
        ModelCheckpoint(
            dirpath=os.path.join(args["output_dir"], 'checkpoints'),
            filename=f'{args["model_type"]}-{{epoch:02d}}-{{val/loss:.4f}}',
            monitor='train/loss_log_MSE',
            mode='min',
            save_top_k=3
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='train/loss_log_MSE',
            patience=args["patience"],
            mode='min',
            verbose=True
        )
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='auto',  # Automatically use GPU if available
        devices=1,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=args["val_check_interval"],
        log_every_n_steps=10,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0
    )

    # Train model
    trainer.fit(model, 
                data_module.train_dataloader(), 
                data_module.val_dataloader(),
                ckpt_path=args.get("resume", None))
    
    # Save final model
    trainer.save_checkpoint(os.path.join(args["output_dir"], f'final_{args["model_type"]}_model.ckpt'))
    
    # Test model
    trainer.test(model, data_module.test_dataloader())
    
    final_model_path = os.path.join(args["output_dir"], f"final_{args['model_type']}_model.ckpt")
    print(f"Training completed! Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()