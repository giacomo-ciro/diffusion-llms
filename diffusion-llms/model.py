import json
import math

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
        input_mask:torch.Tensor=None,
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

        if targets:
            if input_mask is not None:
                B, seq_len, vocab_size = logits.shape

                # Reshape to (B*seq_len, vocab_size)
                logits_flat = logits.view(-1, vocab_size)
                
                # Reshape  to (B*seq_len)
                targets_flat = targets.view(-1)
                mask_flat = input_mask.view(-1)
                
                # Select the logits and targets for masked tokens only
                masked_indices = torch.nonzero(mask_flat, as_tuple=True)[0]
                masked_logits = logits_flat[masked_indices]
                masked_targets = targets_flat[masked_indices]
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
            else:
                loss = torch.nn.functional.cross_entropy(
                    input = logits.view(-1, logits.size(-1)), # (B*context_length, vocab_size)
                    target = targets.reshape(-1),   # (B, context_length) -> (B*context_length,)
                )
        else:
            loss = None

        return logits, loss

    def training_step(self, batch, batch_idx):
        
        # Read in batch
        input_ids, targets, input_mask = batch        
        input_ids = input_ids.cpu().to(torch.int64).to(self.device) # (B, context_length)
        targets = targets.cpu().to(torch.int64).to(self.device) # (B, context_length)

        # Prepare the attention mask:
        B, context_length = input_ids.shape
        if self.config["pipeline"] == "diffusion":
            p = self.annealing_schedule[self.global_step] if self.global_step < len(self.annealing_schedule) else 1.0
            attn_mask = get_annealing_mask(context_length, B, p).to(input_ids.device)
        else:
            attn_mask = get_causal_mask(B, context_length)

        # Forward pass
        logits, loss = self.forward(input_ids, targets, input_mask, attn_mask)
        
        # Logging
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )
        self.log(
            "train/learning_rate",
            self.optimizers().param_groups[0]['lr'],
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        
        # Read in batch
        input_ids, targets, input_mask = batch
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
        logits, loss = self.forward(input_ids, targets, input_mask, attn_mask)
        
        # Logging
        self.log(
            "valid/loss",
            loss,
            on_step=False,  # Logs the metric at the current step
            on_epoch=True,  # Automatically accumulates and logs at the end of the epoch
            prog_bar=True,  # Log to the progbar
        )
        
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
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using diffusion process.
        
        Args:
            input_ids: Starting token sequence (list or tensor)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        """

        assert isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2
        assert temperature > 0
        
        self.eval()
        input_ids = input_ids.to(self.device)

        # Get dimensions
        B = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        
        # Start with fully masked sequence for the new tokens
        seq_len = prompt_len + max_new_tokens
        x = torch.full((B, seq_len), self.config["mask_id"], dtype=torch.long, device=self.device)
        
        # (id, ..., id, msk, ..., msk)
        # [B, seq_len]
        x[:, :prompt_len] = input_ids
        
        # Number of diffusion steps
        num_steps = self.config.get("diffusion_steps", max(64, max_new_tokens // 4)) 
        
        # Number of tokens to be denoised at each step
        n_tokens_per_step = math.ceil(1 / num_steps * max_new_tokens)
        
        # Full attention mask for inference
        attn_mask = get_annealing_mask(seq_len, B, 1.0).to(x.device)
        
        # Gradually unmask tokens
        for step in range(num_steps, 0, -1):
            
            # Get masked positions
            mask = (x == self.config["mask_id"])
            
            # Exit when generation is completed
            if not any(mask):
                break
            
            # Forward pass to get predictions
            with torch.no_grad():
                logits, _ = self.forward(
                    input_ids=x,
                    targets=None,
                    input_mask=None,
                    attn_mask=attn_mask
                )
            
            # Apply temperature and optional top-k sampling
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(
                    input = logits,
                    k = min(top_k, logits.size(-1)),
                    dim=-1
                )
                logits[logits < v[:, :, [-1]]] = -float('Inf')
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample from the distribution
            next_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)), 
                num_samples=1
            ).view(B, seq_len)
            
            # Right shift
            # [0, 1, 2, 3] -> [0, 0, 1, 2]
            next_tokens = torch.cat(
                [next_tokens[:,0:1], next_tokens[:, :-1]],
                 dim=1
            )

            # Compute entropy for each position (along the vocabulary)
            entropy = -torch.sum(
                probs * torch.log2(probs),
                axis = -1
            )

            # Deal with right shift in the entropy computation
            # probs at position i refer to (i+1)-th token in the output
            entropy = torch.cat(
                [entropy[:, 0:1], entropy[:, :-1]],
                 dim=1
            )
            # Set tokens not to be predicted at +inf
            entropy[~mask] = torch.inf

            # Get the most confident predictions
            # (lowest entropy, lowest perplexity)
            idx_to_denoise = torch.topk(
                entropy,
                k=n_tokens_per_step,
                dim=-1,
                largest=False
            )

            # Transform indices to mask
            step_mask = torch.zeros_like(mask)
            step_mask.scatter_(1, idx_to_denoise, True)
            
            # Only predict those
            mask = mask & step_mask
            
            # Update the tokens that were masked
            x = torch.where(mask, next_tokens, x)
            
        return x

    def test_eos_prediction(self, prompt_tokens, eos_token_id, pad_token_id, num_samples=100):
        """
        Test how well the model predicts EOS tokens at the first diffusion step.
        
        Args:
            prompt_tokens: Token IDs of the prompt
            eos_token_id: ID of the EOS token
            pad_token_id: ID of the padding token
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with metrics about EOS prediction accuracy
        """
        eos_predictions = []
        eos_confidences = []
        
        for _ in range(num_samples):
            # Create a sequence with prompt + all masks
            B = 1  # Batch size of 1 for evaluation
            seq_len = self.config["context_length"]
            prompt_len = len(prompt_tokens)
            
            # Create input sequence with prompt followed by masks
            input_ids = torch.full((B, seq_len), self.config["mask_id"], dtype=torch.long, device=self.device)
            input_ids[0, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)
            
            # Create mask for prediction (only predict masked positions)
            mask = torch.zeros((B, seq_len), dtype=torch.bool, device=self.device)
            mask[0, prompt_len:] = True
            
            # Make prediction
            attn_mask = get_annealing_mask(seq_len, B, 1.0).to(self.device)  # Full attention for testing
            logits, _ = self.forward(input_ids, input_ids, mask, attn_mask)
            
            # Get prediction probabilities for masked positions
            probs = torch.softmax(logits[0, prompt_len:], dim=-1)
            
            # Get probability of EOS token at each position
            eos_probs = probs[:, eos_token_id].cpu().numpy()
            
            # Check if EOS has highest probability at any position
            predicted_tokens = torch.argmax(logits[0, prompt_len:], dim=-1).cpu().numpy()
            has_eos = eos_token_id in predicted_tokens
            eos_pos = np.argmax(eos_probs) if has_eos else -1
            
            eos_predictions.append(has_eos)
            eos_confidences.append(np.max(eos_probs))
        
        # Calculate metrics
        eos_prediction_rate = sum(eos_predictions) / num_samples
        avg_confidence = sum(eos_confidences) / num_samples
        
        return {
            "eos_prediction_rate": eos_prediction_rate,
            "avg_eos_confidence": avg_confidence,
            "eos_confidences": eos_confidences
        }