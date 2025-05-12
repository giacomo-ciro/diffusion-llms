import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from pytorch_lightning.callbacks import EarlyStopping
from transformers import AutoModel, AutoTokenizer

class LLadaLightning(pl.LightningModule):
    def __init__(
        self, 
        pretrained_model_name="GSAI-ML/LLaDA-8B-Instruct", 
        context_length=4096,
        learning_rate=1e-5,
        pos_weight=10.0  # For imbalanced classification
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        hidden_size = self.backbone.config.hidden_size
        
        # Classification head with balanced loss
        self.classifier = nn.Linear(hidden_size, 1)
        self.pos_weight = torch.tensor([pos_weight])
        
        # Regression head with normalized outputs
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Outputs between 0-1
        )
        
        # Full sequence regressor
        self.full_regressor = nn.Sequential(
            nn.Linear(hidden_size * context_length, 1),
            nn.Sigmoid()  # Outputs between 0-1
        )
        
        self.context_length = context_length
        self.learning_rate = learning_rate
    
    def forward_backbone(self, input_ids, attention_mask=None):
        # Get hidden states
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        
        # Get last layer hidden states
        last_hidden = outputs.hidden_states[-1]
        
        # Mean pooling for sequence representation
        if attention_mask is not None:
            # Create mask for mean pooling
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_hidden = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.sum(mask, dim=1)
            pooled = sum_hidden / sum_mask
        else:
            pooled = torch.mean(last_hidden, dim=1)
        
        return last_hidden, pooled
    
    def shared_step(self, batch, batch_idx, step_type="train"):
        input_ids = batch["input_ids"]
        eos_labels = batch["eos_labels"].float()
        true_length = batch["true_length"]
        
        # Normalize true_length to 0-1 range
        normalized_true_length = true_length.float() / self.context_length
        
        # Get hidden states
        last_hidden, pooled = self.forward_backbone(input_ids)
        
        # Classification (token-level)
        logits = self.classifier(last_hidden).squeeze(-1)
        loss_clf = F.binary_cross_entropy_with_logits(
            logits, eos_labels, 
            pos_weight=self.pos_weight.to(eos_labels.device)
        )
        
        # Regression (sequence-level)
        preds = self.regressor(pooled).squeeze(-1)
        loss_reg = F.mse_loss(preds, normalized_true_length)
        
        # Full regression
        preds_full = self.full_regressor(last_hidden.view(last_hidden.size(0), -1)).squeeze(-1)
        loss_reg_full = F.mse_loss(preds_full, normalized_true_length)
    
        
        # Log metrics
        self.log(f"{step_type}/loss_clf", loss_clf, prog_bar=True)
        self.log(f"{step_type}/loss_reg", loss_reg, prog_bar=True)
        self.log(f"{step_type}/loss_reg_full", loss_reg_full, prog_bar=True)
        
        # Calculate accuracy for classification
        with torch.no_grad():
            pred_labels = (torch.sigmoid(logits) > 0.5).float()
            acc = (pred_labels == eos_labels).float().mean()
            self.log(f"{step_type}/acc", acc, prog_bar=True)
        
        return loss_clf, loss_reg, loss_reg_full
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        # Separate optimizers for different components
        optimizer_clf = AdamW(self.classifier.parameters(), lr=self.learning_rate)
        optimizer_reg = AdamW(self.regressor.parameters(), lr=self.learning_rate)
        optimizer_reg_full = AdamW(self.full_regressor.parameters(), lr=self.learning_rate)
        
        return [optimizer_clf, optimizer_reg, optimizer_reg_full]

# Main training function
def train_llada_lightning():
    # Load data
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    datamodule = DataModule(config, tokenizer)
    
    # Calculate positive weight for imbalanced dataset
    # (This would require one pass through dataset, could be pre-calculated)
    pos_weight = 10.0  # Default value, adjust based on dataset
    
    # Create model
    model = LLadaLightning(
        pretrained_model_name="GSAI-ML/LLaDA-8B-Instruct",
        context_length=config["context_length"],
        learning_rate=1e-5,
        pos_weight=pos_weight
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode='min'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[early_stop_callback],
        gpus=1 if torch.cuda.is_available() else 0,
        log_every_n_steps=10,
        val_check_interval=0.1  # Validate every 10% of training
    )
    
    # Train model
    trainer.fit(model, datamodule)
    
    return model