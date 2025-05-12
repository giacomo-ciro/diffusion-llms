import os
from tqdm import tqdm

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, root_mean_squared_error, roc_auc_score
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from diffusion_llms.dataloader.llada_2 import DataModule 


from diffusion_llms.input_helper import get_config

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stop = False

    def should_stop(self, current):
        if self.best is None:
            self.best = current
            return False

        improvement = (current < self.best - self.min_delta) if self.mode == 'min' else (current > self.best + self.min_delta)

        if improvement:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop


def save_checkpoint(model, optimizers, epoch, step, metrics, checkpoint_dir):
    """Save model checkpoint to file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_clf_state_dict': optimizers['clf'].state_dict(),
        'optimizer_reg_state_dict': optimizers['reg'].state_dict(),
        'optimizer_reg_full_state_dict': optimizers['reg_full'].state_dict(),
        'best_metrics': {
            'clf': metrics['best_clf'],
            'reg': metrics['best_reg'],
            'reg_full': metrics['best_reg_full']
        }
    }
    
    torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch{epoch}_step{step}.pt")
    # Also save a "best" checkpoint if this is the best model so far
    if metrics['is_best']:
        torch.save(checkpoint, f"{checkpoint_dir}/best_model.pt")
    
    print(f"Checkpoint saved at epoch {epoch}, step {step}")
    wandb.log({"checkpoint_saved": epoch})


def load_checkpoint(model, optimizers, checkpoint_path):
    """Load model checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizers['clf'].load_state_dict(checkpoint['optimizer_clf_state_dict'])
    optimizers['reg'].load_state_dict(checkpoint['optimizer_reg_state_dict'])
    optimizers['reg_full'].load_state_dict(checkpoint['optimizer_reg_full_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    return checkpoint


def init_wandb(config, project_name="diffusion-llms"):
    wandb.init(
        project=project_name,
        name=config["run_name"] if config["run_name"] else "variable-length-llada",
        config=config if config is not None else {},
    )

def collate_fn(batch):
    # Default collate: tensors already uniformly sized
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "eos_labels": torch.stack([item["eos_labels"] for item in batch]),
        "true_length": torch.stack([item["true_length"] for item in batch]),
    }


class VarLenLLada(nn.Module):
    def __init__(
        self, pretrained_model_name="GSAI-ML/LLaDA-8B-Instruct", context_length=4096
    ):
        super().__init__()
        # Backbone transformer
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        # Freeze backbone to reuse hidden states without recomputing graph
        for param in self.backbone.parameters():
            param.requires_grad = False
        hidden_size = self.backbone.config.hidden_size

        # Classification head: pad vs non-pad
        self.classifier = nn.Linear(hidden_size, 1)
        # Regression head: predict sequence length
        self.regressor = nn.Linear(hidden_size, 1)  # uses pooled output

        self.full_regressor = nn.Linear(
            hidden_size * context_length, 1
        )  # uses full hidden states

    def forward_backbone(self, input_ids, attention_mask):
        # Compute hidden states once
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled = outputs.pooler_output  # (batch, hidden)
        return last_hidden, pooled

    def classify(self, last_hidden):
        # Per-token logits
        logits = self.classifier(last_hidden).squeeze(-1)  # (batch, seq_len)
        return logits

    def regress(self, pooled):
        # Sequence length prediction
        pred = self.regressor(pooled).squeeze(-1)  # (batch,)
        return pred

    def regress_no_pool(self, last_hidden):
        # Sequence length prediction
        pred = self.full_regressor(last_hidden.view(last_hidden.size(0), -1))
        return pred.squeeze(-1)  # (batch,)


@torch.no_grad()
def eval_epoch(model, dataloader, device, best_metrics=None):
    model.eval()

    loss_fn_clf = nn.BCEWithLogitsLoss()
    loss_fn_reg = nn.MSELoss()

    all_logits = []
    all_eos_labels = []

    all_preds = []
    all_preds_full = []
    all_true_lengths = []

    total_loss_cat = 0.0
    total_loss_reg = 0.0
    total_loss_reg_full = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        eos_labels = batch["eos_labels"].float().to(device)
        true_length = batch["true_length"].to(device)

        last_hidden, pooled = model.forward_backbone(input_ids, attention_mask) 

        logits = model.classify(last_hidden) # [batch, seq_len]
        loss_clf = loss_fn_clf(logits, eos_labels)

        preds = model.regress(pooled)
        loss_reg = loss_fn_reg(preds, true_length)

        preds_full = model.regress_no_pool(last_hidden)
        loss_reg_full = loss_fn_reg(preds_full, true_length)

        total_loss_cat += loss_clf.item()
        total_loss_reg += loss_reg.item()
        total_loss_reg_full += loss_reg_full.item()

        all_logits.append(logits.detach().cpu())
        all_eos_labels.append(eos_labels.detach().cpu())

        all_preds.append(preds.detach().cpu())
        all_preds_full.append(preds_full.detach().cpu())
        all_true_lengths.append(true_length.detach().cpu())

        num_batches += 1

    # Flatten across batches
    flat_logits = torch.cat(all_logits).flatten()
    flat_labels = torch.cat(all_eos_labels).flatten()

    bin_preds = (torch.sigmoid(flat_logits) > 0.5).int()
    clf_acc = accuracy_score(flat_labels.int(), bin_preds)
    clf_auc = roc_auc_score(flat_labels.numpy(), torch.sigmoid(flat_logits).numpy())

    true_lengths = torch.cat(all_true_lengths)
    reg_preds = torch.cat(all_preds)
    reg_preds_full = torch.cat(all_preds_full)

    reg_rmse = root_mean_squared_error(true_lengths, reg_preds)
    reg_rmse_full = root_mean_squared_error(true_lengths, reg_preds_full)

    avg_loss_clf = total_loss_cat / num_batches
    avg_loss_reg = total_loss_reg / num_batches
    avg_loss_reg_full = total_loss_reg_full / num_batches

    wandb.log(
        {
            "val/loss_clf": avg_loss_clf,
            "val/acc_clf": clf_acc,
            "val/auc_clf": clf_auc,
            "val/loss_reg": avg_loss_reg,
            "val/rmse_reg": reg_rmse,
            "val/loss_reg_full": avg_loss_reg_full,
            "val/rmse_reg_full": reg_rmse_full,
        }
    )

    print(
        f"[VAL] Clf Loss: {avg_loss_clf:.4f}, Acc: {clf_acc:.4f}, AUC: {clf_auc:.4f} | "
        f"Reg Loss: {avg_loss_reg:.4f}, RMSE: {reg_rmse:.2f} | "
        f"RegFull Loss: {avg_loss_reg_full:.4f}, RMSE: {reg_rmse_full:.2f}"
    )
    
    # Check if this is the best model so far
    is_best = False
    if best_metrics is not None:
        if avg_loss_clf < best_metrics['best_clf']:
            best_metrics['best_clf'] = avg_loss_clf
            is_best = True
        if avg_loss_reg < best_metrics['best_reg']:
            best_metrics['best_reg'] = avg_loss_reg
            is_best = True
        if avg_loss_reg_full < best_metrics['best_reg_full']:
            best_metrics['best_reg_full'] = avg_loss_reg_full
            is_best = True
    
    return {
        'loss_clf': avg_loss_clf,
        'loss_reg': avg_loss_reg,
        'loss_reg_full': avg_loss_reg_full,
        'is_best': is_best
    }


def main():
    # Settings
    pretrained_model = "GSAI-ML/LLaDA-8B-Instruct"
    lr = 1e-5
    batch_size = 8
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    # Create checkpoint directory
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    init_wandb(config)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    # DataModule
    datamodule = DataModule(config, tokenizer)
    datamodule.setup()

    # Model & optimizers
    model = VarLenLLada(pretrained_model).to(device)
    optimizer_clf = AdamW(model.classifier.parameters(), lr=lr)
    optimizer_reg = AdamW(model.regressor.parameters(), lr=lr)
    optimizer_reg_full = AdamW(model.full_regressor.parameters(), lr=lr)
    
    # Store optimizers in a dictionary for easier checkpointing
    optimizers = {
        'clf': optimizer_clf,
        'reg': optimizer_reg,
        'reg_full': optimizer_reg_full
    }

    # Initialize best metrics and start epoch/step
    best_metrics = {
        'best_clf': float('inf'),
        'best_reg': float('inf'),
        'best_reg_full': float('inf')
    }
    start_epoch = 1
    global_step = 0

    # Resume from checkpoint if specified
    if config["resume_from_checkpoint"]:
        checkpoint = load_checkpoint(model, optimizers, config["resume_from_checkpoint"])
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['step']
            best_metrics = checkpoint['best_metrics']
            print(f"Resuming from epoch {start_epoch}, step {global_step}")

    stopper_clf = EarlyStopper(patience=3, mode='min', min_delta=0.001)
    stopper_reg = EarlyStopper(patience=3, mode='min', min_delta=0.001)
    stopper_reg_full = EarlyStopper(patience=3, mode='min', min_delta=0.001)

    stop_clf = stop_reg = stop_reg_full = False

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        val_check_steps = config.get("val_check_steps", 100)
        checkpoint_steps = config.get("checkpoint_steps", 200)

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        model.train()
        loss_fn_clf = nn.BCEWithLogitsLoss()
        loss_fn_reg = nn.MSELoss()

        total_loss_cat = 0.0
        total_loss_reg = 0.0
        total_loss_reg_full = 0.0
        num_batches = 0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
        for idx, batch in train_pbar:
            global_step += 1
            
            # Validation checkpoint
            if global_step % val_check_steps == 0:
                val_metrics = eval_epoch(model, val_loader, device, best_metrics)
                
                # Save checkpoint if this is the best model so far
                if val_metrics['is_best']:
                    print("New best model found!")
                    save_checkpoint(
                        model, 
                        optimizers,
                        epoch,
                        global_step,
                        {'best_clf': best_metrics['best_clf'], 
                         'best_reg': best_metrics['best_reg'], 
                         'best_reg_full': best_metrics['best_reg_full'],
                         'is_best': True},
                        config["checkpoint_dir"]
                    )
                
                # Update early stopping flags
                if not stop_clf:
                    stop_clf = stopper_clf.should_stop(val_metrics['loss_clf'])
                if not stop_reg:
                    stop_reg = stopper_reg.should_stop(val_metrics['loss_reg'])
                if not stop_reg_full:
                    stop_reg_full = stopper_reg_full.should_stop(val_metrics['loss_reg_full'])
                
                # Log validation metrics in tqdm
                train_pbar.set_postfix({
                    'val/loss_clf': f"{val_metrics['loss_clf']:.4f}",
                    'val/loss_reg': f"{val_metrics['loss_reg']:.4f}",
                    'val/loss_reg_full': f"{val_metrics['loss_reg_full']:.4f}"
                })
            
            # Regular checkpoint
            if global_step % checkpoint_steps == 0:
                save_checkpoint(
                    model, 
                    optimizers,
                    epoch,
                    global_step,
                    {'best_clf': best_metrics['best_clf'], 
                     'best_reg': best_metrics['best_reg'], 
                     'best_reg_full': best_metrics['best_reg_full'],
                     'is_best': False},
                    config["checkpoint_dir"]
                )

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            eos_labels = batch["eos_labels"].float().to(device)
            true_length = batch["true_length"].to(device)

            # Compute hidden states WITH gradients
            last_hidden, pooled = model.forward_backbone(input_ids, attention_mask)

            last_hidden = last_hidden.detach()  # Detach to avoid recomputing graph
            pooled = pooled.detach()  # Detach to avoid recomputing graph

            if not stop_clf:
                # Classification head update
                optimizer_clf.zero_grad()
                logits = model.classify(last_hidden)
                loss_clf = loss_fn_clf(logits, eos_labels)
                loss_clf.backward()
                optimizer_clf.step()
                total_loss_cat += loss_clf.item()
                wandb.log({
                    "train/loss_clf": loss_clf.item(),
                    "step": global_step
                })
                

            if not stop_reg:
                # Regression head update
                optimizer_reg.zero_grad()
                preds = model.regress(pooled)
                loss_reg = loss_fn_reg(preds, true_length)
                loss_reg.backward()
                optimizer_reg.step()
                total_loss_reg += loss_reg.item()
                wandb.log({
                    "train/loss_reg": loss_reg.item(),
                    "step": global_step
                })

            if not stop_reg_full:
                # Full regression head update
                optimizer_reg_full.zero_grad()
                preds_full = model.regress_no_pool(last_hidden)
                loss_reg_full = loss_fn_reg(preds_full, true_length)
                loss_reg_full.backward()
                optimizer_reg_full.step()
                total_loss_reg_full += loss_reg_full.item()
                wandb.log({
                    "train/loss_reg_full": loss_reg_full.item(),
                    "step": global_step
                })
            
            # Log training metrics in tqdm
            train_pbar.set_postfix({
                'train/loss_clf': f"{total_loss_cat / (num_batches + 1):.4f}",
                'train/loss_reg': f"{total_loss_reg / (num_batches + 1):.4f}",
                'train/loss_reg_full': f"{total_loss_reg_full / (num_batches + 1):.4f}"
            })

            if stop_clf and stop_reg and stop_reg_full:
                print("Early stopping triggered.")
                # Save final checkpoint before stopping
                save_checkpoint(
                    model, 
                    optimizers,
                    epoch,
                    global_step,
                    {'best_clf': best_metrics['best_clf'], 
                     'best_reg': best_metrics['best_reg'], 
                     'best_reg_full': best_metrics['best_reg_full'],
                     'is_best': False},
                    config["checkpoint_dir"]
                )
                break

            num_batches += 1
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(
            model, 
            optimizers,
            epoch,
            global_step,
            {'best_clf': best_metrics['best_clf'], 
             'best_reg': best_metrics['best_reg'], 
             'best_reg_full': best_metrics['best_reg_full'],
             'is_best': False},
            config["checkpoint_dir"]
        )
        
        if stop_clf and stop_reg and stop_reg_full:
            print("Early stopping triggered. Training complete.")
            break


if __name__ == "__main__":
    main()