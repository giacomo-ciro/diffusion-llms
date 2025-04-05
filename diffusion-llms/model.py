import json
import math
import os

import torch
import lightning as pl
from torch.optim.lr_scheduler import _LRScheduler
from gpt2 import GPT, GPTConfig

class GPT2(pl.LightningModule):
    """
    Wraps the GPT class from Andrej Karpathy NanoGPT, and adapts to Lightning framework.
    """

    def __init__(
        self,
        config_path,
    ):
        super().__init__()

        # Load configuration file
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        # Get parameters
        self.weight_decay = self.config["weight_decay"]
        self.betas = self.config["betas"]

        # Init the model
        # Pre-trained from hugging face
        if self.config["init_from"].startswith("gpt2"):
            self.gpt2 = GPT.from_pretrained(
                model_type = self.config["init_from"],
                override_args = None
            )
            print(f"Loaded GPT-2 from {self.config["init_from"]}.")
        
        # Our saved checkpoint
        elif self.config["init_from"].endswith(".ckpt"):
            
            # Path/to/.ckpt
            ckpt_config_path = os.path.join(
                os.path.dirname(self.config["init_from"]),
                "config.json"
            )
            assert os.path.exists(ckpt_config_path), "ckpt folder must contain config.json"
            
            # Path/to/config.json
            with open(os.path.join(ckpt_config_path), "r") as f:
                ckpt_config = json.load(f)
            
            # Instance with same params
            gptconfig = GPTConfig(
                block_size=ckpt_config["context_length"],
                n_embd=ckpt_config["n_embd"],
                n_layer=ckpt_config["n_layer"],
                n_head=ckpt_config["n_head"],
            )
            self.gpt2 = GPT(gptconfig)
            
            # Load weights
            checkpoint = torch.load(self.config["init_from"], map_location='cpu')
            assert "state_dict" in checkpoint.keys()
            state_dict = {k.replace("gpt2.", ""):v for k,v in checkpoint["state_dict"].items()}
            self.gpt2.load_state_dict(state_dict)
            print(f"Loaded GPT-2 from {self.config["init_from"]}.")
        
        # A new version from scratch
        else:
            gptconfig = GPTConfig(
                block_size=self.config["context_length"],
                n_embd=self.config["n_embd"],
                n_layer=self.config["n_layer"],
                n_head=self.config["n_head"],
            )
            self.gpt2 = GPT(gptconfig)
            print("Initialized new GPT-2.")

    def forward(
        self,
        idx,
        targets= None
    ) -> dict:
        # GPT().forward(idx, targets=None) returns logits, loss
        # if target is not provided, it returns logits only for 
        # the last position and no loss
        logits, loss = self.gpt2(idx, targets)

        return logits, loss

    def step(self, batch, batch_idx):
        
        # Read in the batch
        # X is token idx, (B, T)
        # y is idx shifted by one, (B, T)
        X, y = batch
        X = X.cpu().to(torch.int64).to(self.device)
        y = y.cpu().to(torch.int64).to(self.device)

        logits, loss = self.forward(X, y)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train/ce", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True, on_step=True, on_epoch=False)
        
        # # Log all parameter group learning rates
        # for i, param_group in enumerate(self.optimizers().param_groups):
        #     self.log(f'learning_rate_group_{i}', param_group['lr'], 
        #              prog_bar=False, on_step=True, on_epoch=False)
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("valid/ce", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.gpt2.configure_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.config["max_lr"],
            betas=self.betas,
            device_type="cuda" if torch.cuda.is_available else "cpu"
        )        

        scheduler = CosineWarmupScheduler(
            optimizer,
            self.config_path
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,  # Check after each step
            },
        }
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if isinstance(idx, list):       # convert to tensor
            idx = torch.tensor(idx)
        if len(idx.shape) == 1:         # add batch dim if missing
            idx = idx.unsqueeze(0)
        return self.gpt2.generate(idx, max_new_tokens, temperature, top_k)

class CosineWarmupScheduler(_LRScheduler):
    """
    Warmup scheduler.
    For the first warmup_steps iterations, progressively grows to max_lr.
    Then gradually decays using cosine annealing to min_lr and keeps until the end. 
    """
    def __init__(
        self, 
        optimizer, 
        config_path,
        last_epoch=-1,
    ):
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.max_lr = self.config["max_lr"]
        self.min_lr = self.config["min_lr"]
        self.warmup_steps=self.config["warmup_steps"]
        self.lr_decay_steps= int(self.config["lr_decay_fraction_n_steps"] * self.config["n_steps"])
        
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        it = self.last_epoch
        
        # 1) linear warmup for warmup_steps steps
        if it < self.warmup_steps:
            return [self.max_lr * (it + 1) / (self.warmup_steps + 1) 
                    for _ in self.base_lrs]
        
        # 2) if it > lr_decay_steps, return min learning rate
        if it > self.lr_decay_steps:
            return [self.min_lr for _ in self.base_lrs]
        
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.lr_decay_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        
        return [
            self.min_lr + coeff * (self.max_lr - self.min_lr) 
            for _ in self.base_lrs
        ]
