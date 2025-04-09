import json

import numpy as np
import torch
import lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from transformers import GPT2LMHeadModel, GPT2Config

class GPT2(pl.LightningModule):
    """
    Wraps the GPT2LMHeadModel class from HuggingFace, and adapts to Lightning framework.
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

        # Init the model
        # Pre-trained from hugging face
        if self.config["init_from"].startswith("gpt2"):
            assert self.config["init_from"] in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
            print(f"Loading pre-trained {self.config["init_from"]}...")
            self.gpt2 = GPT2LMHeadModel.from_pretrained(
                pretrained_model_name_or_path = self.config["init_from"]
            )
        
        # A new version from scratch
        else:
            print("Initializing new GPT-2...")
            gptconfig = GPT2Config(
                n_positions=self.config["context_length"],
                n_embd=self.config["n_embd"],
                n_layer=self.config["n_layer"],
                n_head=self.config["n_head"],
            )
            self.gpt2 = GPT2LMHeadModel(gptconfig)
        
        # For the annealing 
        # At each step i, the entries in the attention mask 
        # will be un-masked with probability annealing_schedule[i]
        # at the last annealing step, all entries will be unmasked
        if self.config["pipeline"] == "diffusion":
            self.annealing_schedule = np.linspace(
                0,
                1,
                self.config["attn_annealing_steps"]
            )
    def forward(
        self,
        input_ids,
        attn_mask=None
    ) -> tuple:
        
        # Logging
        non_zero_mask = attn_mask.sum().item()
        self.log("non_zero_mask", non_zero_mask)

        # If labels provided, are automatically shifted, 
        # hence set labels = input_ids
        logits = self.gpt2.forward(
            input_ids = input_ids,
            attn_mask=attn_mask
        ).logits

        # TODO: comupute loss 
        loss = 0.0

        return non_zero_mask

    def training_step(self, batch, batch_idx):
        
        # Read in batch
        X, y = batch
        X = X.cpu().to(torch.int64).to(self.device)
        y = y.cpu().to(torch.int64).to(self.device)

        # Prepare the attention mask:
        attn_mask = torch.tril(torch.ones(X.shape[1], X.shape[1]), device = X.device).to(torch.bool)
        if self.config["pipeline"] == "diffusion": 
            p = self.annealing_schedule[self.global_step]
            attn_unmask = torch.rand(size=(X.shape[1], X.shape[1]), device = X.device) <= p
            attn_mask = attn_mask | attn_unmask

        # Forward pass
        loss = self.forward(batch, batch_idx)
        
        # Logging
        self.log("train/ce", loss, on_step=True, on_epoch=False, prog_bar=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        
        # Read in batch
        X, y = batch
        X = X.cpu().to(torch.int64).to(self.device)
        y = y.cpu().to(torch.int64).to(self.device)

        # Can see everything when validation
        attn_mask = torch.ones(X.shape[1], X.shape[1]).to(X.device).to(torch.bool)
        
        # Forward pass
        loss = self.forward(X, attn_mask=attn_mask)
        
        # Logging
        self.log("valid/ce", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        # create optim groups. Any parameters that is 2D will be weight decayed, 
        # otherwise no. i.e. all weight tensors in matmuls + embeddings decay, 
        # all biases and layernorms don't.
        # Layer normalization parameters and biases already have limited
        # degrees of freedom, so they don't need regularization as much 
        # as weight matrices do. This approach has been shown empirically 
        # to lead to better convergence and model performance.

        # Divide params in decay and non-decay
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {
                'params': decay_params,
                'weight_decay': self.config["weight_decay"]
            },
            {
                'params': nodecay_params,
                'weight_decay': 0.0
            }
        ]

        # Optimizer
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            learning_rate=self.config["max_lr"],
            betas = self.config["betas"],
            fused=True
        )        

        # The 1cycle policy (warm-up + annealing)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config["max_lr"],
            total_steps=self.config["n_steps"],
            pct_start=self.config["pct_start"],  # Warm-up percentage of total steps
            div_factor=self.config["div_factor"],  # Determines initial_lr = max_lr / div_factor
            final_div_factor=self.config["final_div_factor"],  # Determines min_lr = initial_lr / final_div_factor
            anneal_strategy="cos",
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
        pass