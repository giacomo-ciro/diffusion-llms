import json

import numpy as np
import torch
import lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from transformers import GPT2LMHeadModel, GPT2Config
from attention_patch import replace_attention_mask
from utils import get_annealing_mask, get_causal_mask

# Modify the HF implementation so 
# the attention mask provided is actually used
replace_attention_mask()

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
        init_from = self.config["init_from"]
        if init_from.startswith("gpt2"):
            assert init_from in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
            print(f"Loading pre-trained {init_from}...")
            self.gpt2 = GPT2LMHeadModel.from_pretrained(
                pretrained_model_name_or_path = init_from
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
        input_ids:torch.Tensor,
        targets:torch.Tensor,
        attn_mask:torch.Tensor=None
    ) -> tuple:
        
        # Logging
        # Store the percentage of the non_zero entries in the mask
        assert len(attn_mask.shape) == 4
        non_zero_mask = attn_mask.sum().item() / attn_mask.numel()
        self.log("non_zero_mask", non_zero_mask, on_step=True, on_epoch=False, prog_bar=True)

        # If labels provided, gpt2 automatically shifts them to 
        # compute the loss (here we do it manually) and returns 
        # a CausalLMOutputWithCrossAttentions object, with 
        # attribute .logits
        logits = self.gpt2.forward(
            input_ids = input_ids,
            attn_mask = attn_mask
        ).logits

        if self.config["pipeline"] == "diffusion":
            # TODO: comupute loss 
            pass
        else:
            loss = torch.nn.functional.cross_entropy(
                input = logits.view(-1, logits.size(-1)), # (B*context_length, vocab_size)
                target = targets.reshape(-1),   # (B, context_length) -> (B*context_length,)
            )

        return logits, loss

    def training_step(self, batch, batch_idx):
        
        # Read in batch
        input_ids, targets = batch        
        input_ids = input_ids.cpu().to(torch.int64).to(self.device) # (B, context_length)
        targets = targets.cpu().to(torch.int64).to(self.device) # (B, context_length)

        # Prepare the attention mask:
        B, context_length = input_ids.shape
        if (self.config["pipeline"] == "diffusion" and
            self.global_step < len(self.annealing_schedule)):
            attn_mask = get_annealing_mask(context_length, B, 1.0).to(input_ids.device)
        else:
            attn_mask = get_causal_mask(B, context_length)

        # Forward pass
        logits, loss = self.forward(input_ids, targets, attn_mask)
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        
        # Read in batch
        input_ids, targets = batch
        input_ids = input_ids.cpu().to(torch.int64).to(self.device)
        targets = targets.cpu().to(torch.int64).to(self.device)

        # Attention mask
        B, context_length = input_ids.shape
        if self.config["pipeline"] == "diffusion":
            # can see everything at validation
            attn_mask = get_annealing_mask(context_length, B, 1.0).to(input_ids.device)
        else:
            attn_mask = get_causal_mask(context_length, B)
        
        # Forward pass
        logits, loss = self.forward(input_ids, targets, attn_mask=attn_mask)
        
        # Logging
        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
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
            lr=self.config["max_lr"],
            betas = self.config["betas"],
            fused=True
        )        

        # The 1cycle policy (warm-up + annealing)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config["max_lr"],
            total_steps=self.config["n_steps"],
            pct_start=self.config["warmup_pct"],  # Warm-up percentage of total steps
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