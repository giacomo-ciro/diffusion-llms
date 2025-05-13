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

# Import the embedded data module
from diffusion_llms.dataloader.llada_from_file import EmbeddedDataModule
from diffusion_llms.input_helper import get_config
torch.set_float32_matmul_precision("medium")


class LLaDaClassifier(nn.Module):
    """Classification head for LLaDA. Takes a single hidden state and outputs a logit."""
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_state):
        # hidden_state: (batch_size, hidden_size)
        return self.classifier(hidden_state).squeeze(-1)

class LLaDaRegressor(nn.Module):
    """Regression head for LLaDA using pooled output"""
    def __init__(self, hidden_size):
        super().__init__()
        self.regressor = nn.Linear(hidden_size, 1)
    
    def forward(self, pooled):
        return self.regressor(pooled).squeeze(-1)

class LLaDaFullRegressor(nn.Module):
    """Regression head for LLaDA using all hidden states"""
    def __init__(self, hidden_size, context_length):
        super().__init__()
        self.full_regressor = nn.Linear(hidden_size * context_length, 1)
    
    def forward(self, hidden_states):
        batch_size = hidden_states.size(0)
        flattened = hidden_states.view(batch_size, -1)
        return self.full_regressor(flattened).squeeze(-1)

class LLaDaTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training a single LLaDA model.
    """
    def __init__(
        self,
        model_type,
        hidden_size,
        context_length,
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
        
        if model_type == "classifier":
            self.model = LLaDaClassifier(hidden_size)
        elif model_type == "regressor":
            self.model = LLaDaRegressor(hidden_size)
        elif model_type == "full_regressor":
            self.model = LLaDaFullRegressor(hidden_size, context_length)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Extract the appropriate data based on model type
        if self.model_type == "classifier":
            x = batch["last_hidden"]
            y = batch["eos_labels"].float()
            # Shuffle the indexes
            idx = torch.randperm(x.size(0))
            x = x[idx]
            y = y[idx]
            
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
            x = batch["pooled"]
            y = batch["true_length"].float()
            
            # Normalize target to 0-1 range
            y_normalized = y / self.context_length
            
            preds = self.model(x)
            loss = F.mse_loss(preds, y_normalized)
            
            # Log metrics
            self.log('train/loss', loss, prog_bar=True)
            
            # Calculate RMSE in original scale
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(preds * self.context_length, y))
                self.log('train/rmse', rmse, prog_bar=True)
                
        elif self.model_type == "full_regressor":
            x = batch["last_hidden"]
            y = batch["true_length"].float()
            
            # Normalize target to 0-1 range
            y_normalized = y / self.context_length
            
            preds = self.model(x)
            loss = F.mse_loss(preds, y_normalized)
            
            # Log metrics
            self.log('train/loss', loss, prog_bar=True)
            
            # Calculate RMSE in original scale
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(preds * self.context_length, y))
                self.log('train/rmse', rmse, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Extract the appropriate data based on model type
        if self.model_type == "classifier":
            x = batch["last_hidden"]
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
            x = batch["pooled"]
            y = batch["true_length"].float()
            
            # Normalize target to 0-1 range
            y_normalized = y / self.context_length
            
            preds = self.model(x)
            loss = F.mse_loss(preds, y_normalized)
            
            # Log metrics
            self.log('val/loss', loss, prog_bar=True)
            
            # Calculate RMSE in original scale
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(preds * self.context_length, y))
                self.log('val/rmse', rmse, prog_bar=True)
                
        elif self.model_type == "full_regressor":
            x = batch["last_hidden"]
            y = batch["true_length"].float()
            
            # Normalize target to 0-1 range
            y_normalized = y / self.context_length
            
            preds = self.model(x)
            loss = F.mse_loss(preds, y_normalized)
            
            # Log metrics
            self.log('val/loss', loss, prog_bar=True)
            
            # Calculate RMSE in original scale
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(preds * self.context_length, y))
                self.log('val/rmse', rmse, prog_bar=True)
        
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        # Same as validation step
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer for the model"""
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)


def main():
    args = get_config()    
    # Create output directory
    os.makedirs(args["output_dir"], exist_ok=True)
    
    # Create config for data module
    
    # Create data module
    data_module = EmbeddedDataModule(
        args, 
        args["embedding_dir"], 
        num_workers=args["num_workers"]
    )
    data_module.setup()
    
    # Create model
    model = LLaDaTrainer(
        model_type=args["model_type"],
        hidden_size=args["hidden_size"],
        context_length=args["context_length"],
        learning_rate=args["learning_rate"],
        pos_weight=args["pos_weight"]
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
            monitor='val/loss',
            mode='min',
            save_top_k=3
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val/loss',
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
        log_every_n_steps=10
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