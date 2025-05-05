import json
import math
import os

import numpy as np
import torch
import lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from transformers import GPT2LMHeadModel, GPT2Config, GenerationConfig
from attention_patch import replace_attention_mask
from utils import get_annealing_mask, get_causal_mask, compute_binary_metrics

# Modify the HF implementation so 
# the attention mask provided is actually used
# JACK: manually inspected the call tree and 
# printed all intermediate attention mask,
# confirmed that without using this patch 
# the provided attention mask is not considered
# --> calling this function is necessary
# to enable custom attention masks !
replace_attention_mask()   

class GPT2(pl.LightningModule):
    """
    Wraps the GPT2LMHeadModel class from HuggingFace, and adapts it to Lightning framework.
    """

    def __init__(
        self,
        config_path,
    ):
        """
        Initializes the GPT2 LightningModule.

        This constructor performs the following steps:
        1. Loads the model configuration from the specified JSON file (`config_path`).
        2. Initializes the underlying GPT-2 model based on the `init_from` setting
           in the configuration. This can be:
           - A standard Hugging Face GPT-2 variant ('gpt2', 'gpt2-medium', etc.).
           - A pre-trained DiffuGPT model ('diffugpt').
           - A new GPT-2 model initialized from scratch (if `init_from` is not
             one of the above or is explicitly set to initialize from scratch).
        3. If the configured pipeline is 'diffusion' and `attn_annealing_steps` > 0,
           it sets up a linear annealing schedule (`self.annealing_schedule`) for
           the attention mask. This schedule determines the probability of unmasking
           attention entries at each training step during the annealing phase.
        4. If a `pad_token_id` is specified in the configuration (expected to be 50257),
           it extends the model's vocabulary and embedding layers to accommodate this
           additional token using the `extend_vocab` method.

        Args:
            config_path (str): Path to the JSON configuration file for the model.
        """
        super().__init__()

        # Load configuration file
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Init the model
        init_from = self.config["init_from"]
        if init_from.startswith("gpt2"):
            print(f"Loading pre-trained {init_from}...")
            self.init_gpt2(pretrained=init_from)
        elif init_from == "diffugpt":
            print("Loading pre-trained DiffuGPT...")
            self.init_diffugpt()
        else:
            print("Initializing new gpt2...")
            self.init_gpt2(pretrained=False)
        
        # For the annealing 
        # At each step i, the entries in the attention mask 
        # will be un-masked with probability annealing_schedule[i]
        # at the last annealing step, all entries will be unmasked
        if self.config["pipeline"] == "diffusion" and self.config["attn_annealing_steps"] > 0:
            self.annealing_schedule = np.linspace(
                0,
                1,
                self.config["attn_annealing_steps"]
            )
            
        # Extend the WTE and LM_HEAD if required
        if self.config["pad_token_id"]:
            assert self.config["pad_token_id"] <= 50257
            if self.config["pad_token_id"] == 50257:
                self.extend_vocab()

        # Initialize a new head for eos prediction
        if self.config["use_pad_head"]:
            # Freeze all parameters
            for param in self.gpt2.parameters():
                param.requires_grad = False
            # wte = self.gpt2.transformer.wte.weight
            
            # # Unfreeze EOS / PAD embedding
            # if self.config["pad_token_id"] == self.config["eos_token_id"]:
            #     wte[-1].requires_grad = True
            # else:
            #     wte[-2].requires_grad = True 
            #     wte[-2].requires_grad = True 
            
            # # Unfreeze mask embedding
            # wte[self.config["mask_id"]] = True
            
            # Init new head to binary predict eos
            self.eos_head = torch.nn.Linear(self.gpt2.lm_head.in_features, 1)

    def init_diffugpt(self):
        """
        Loads the pre-trained DiffuGPT:
            1. Initializes a new gpt2 from scratch
            2. Downloads DiffuGPT weights from HF
            3. Renames it accordingly
            4. Loads the weights and checks match (wte & lm_head are tied)
            5. Adds a new row in the embedding matrix
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        # Initialize a new gpt2-2 model
        gptconfig = GPT2Config(
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        )
        self.gpt2 = GPT2LMHeadModel(gptconfig)

        # Checks
        wte_before = self.gpt2.lm_head.weight.clone()
        lm_head_before = self.gpt2.transformer.wte.weight.clone()
        assert torch.all(wte_before == lm_head_before)

        # Download the weights
        path_to_safetensor = hf_hub_download(
            repo_id="diffusionfamily/diffugpt-s",
            filename="model.safetensors"
        )        
        model_weights = load_file(path_to_safetensor)
        
        # Adjust names
        state_dict = {k.replace("denoise_model", "transformer"):v for k,v in model_weights.items()}
        state_dict["transformer.wte.weight"] = state_dict["embed_tokens.weight"].clone()
        
        # LM Head and WTE are tied
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
        del state_dict["embed_tokens.weight"]
        
        # Load weights
        incompatible_keys = self.gpt2.load_state_dict(state_dict, strict=False)
        print(incompatible_keys)

        # Checks
        wte_after = self.gpt2.lm_head.weight.clone()
        lm_head_after = self.gpt2.transformer.wte.weight.clone()
        assert not torch.allclose(wte_before, wte_after)
        assert not torch.allclose(lm_head_before, lm_head_after)
        assert torch.allclose(wte_after, lm_head_after)

    def init_gpt2(
            self,
            pretrained: str=None,
        ):
        """
        Initializes a gpt2 model from scratch or from HF checkpoint.
        """

        if pretrained:
            assert pretrained in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
            self.gpt2 = GPT2LMHeadModel.from_pretrained(
                pretrained_model_name_or_path = pretrained
            )
        else:
            gptconfig = GPT2Config(
                n_positions=self.config["context_length"],
                n_embd=self.config["n_embd"],
                n_layer=self.config["n_layer"],
                n_head=self.config["n_head"],
            )
            self.gpt2 = GPT2LMHeadModel(gptconfig)

    def extend_vocab(self):
        """
        Adapts the wte and lm_head to handle an additional token with id=50257
        """
        # Create new wte & lm_head with additional token
        old_wte = self.gpt2.transformer.wte
        new_wte = torch.nn.Embedding(
            old_wte.weight.shape[0]+1,
            old_wte.weight.shape[1]
        )
        old_lm_head = self.gpt2.lm_head
        new_lm_head = torch.nn.Linear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features+1,
            bias=False
        ) 
        # Copy the old weights, init new to the mean
        with torch.no_grad():
            new_wte.weight[:-1] = old_wte.weight
            new_wte.weight[-1] = torch.mean(old_wte.weight, axis = 0)
            # # Copy weights to the new lm_head
            # new_lm_head.weight[:-1] = old_lm_head.weight
            # new_lm_head.weight[-1] = torch.mean(old_lm_head.weight, axis=0)
        
        # Assign to the model
        self.gpt2.transformer.wte = new_wte
        self.gpt2.lm_head = new_lm_head

        # Tie the weights
        self.gpt2.lm_head.weight = self.gpt2.transformer.wte.weight

        # Clear old
        del old_wte
        del old_lm_head
        torch.cuda.empty_cache()  # If using GPU
    
    def forward(
        self,
        input_ids:torch.Tensor,
        targets:torch.Tensor,
        input_mask:torch.Tensor,
        attention_mask:torch.Tensor      # [B, 1, context_length, context_length]    to be broadcasted to keys, querys, values
    ) -> tuple:
        """
        Computes the logits for the given input_ids. 
        If targets are provided computes the loss.
        It requires attention_mask to be always specified.

        Params:
            - input_ids (torch.Tensor[torch.int64]): the input tokens id. Of shape [B, seq_len].
            - targets (torch.Tensor[torch.int64]): the target tokens to be predicted. Of shape [B, seq_len].
            - input_mask (torch.Tensor[bool]): which tokens in the input are masked. Of shape [B, seq_len]. 
            (b, i) = True means the i-th token in the b-th batch is masked 
            - attention_mask (torch.Tensor[bool]): the broadcastable attention mask of shape [B, 1, seq_len, seq_len]. 
            (b, 0, i, j) = True means the i-th token in the b-th batch can attent to
            the j-th token in the b-th batch. 
        """

        # -- Deal with the Attention Matrix
        # 1. Log the percentage of the non_zero entries in the mask
        assert len(attention_mask.shape) == 4
        attention_mask = attention_mask.to(self.device)

        # # 2. Update attn_mask to prevent tokens to look at masked ones
        # B, _, _, seq_len = attention_mask.shape
        # # [B, seq_len]    -> masked_tokens[b, i] = True means the i-th token of the b-th batch is masked
        # masked_tokens = torch.eq(input_ids, self.config["mask_id"])
        # # Sanity check
        # if input_mask is not None:
        #     assert torch.allclose(masked_tokens, input_mask) or torch.all(input_mask)
        # # [B, 1, 1, seq_len]
        # masked_tokens = masked_tokens[:, None, None, :]
        # # [B, 1, seq_len, seq_len]
        # masked_tokens = masked_tokens.expand(-1, -1, seq_len, -1)
        # # Silence the masked tokens (set to false in the attention mask)
        # attention_mask = attention_mask * (~masked_tokens)
        # # 3. Activate back the diagonal (tokens can attend to themselves)
        # # [seq_len, seq_len]
        # diagonal_mask = torch.eye(seq_len, dtype=torch.bool, device=self.device)
        # # [1, 1, seq_len, seq_len]
        # diagonal_mask = diagonal_mask[None, None, :, :]
        # # [B, 1, seq_len, seq_len]
        # diagonal_mask = diagonal_mask.expand(B, 1, seq_len, seq_len)
        # attention_mask = attention_mask | diagonal_mask

        # Forward pass
        transformer_output = self.gpt2.transformer(
            input_ids = input_ids,
            attention_mask = attention_mask
        ).last_hidden_state

        # Get logits
        # When only eos head
        if self.config["use_pad_head"]:
            logits = self.eos_head(
                transformer_output
            )
        # Entire model
        else:
            logits = self.gpt2.lm_head(
                transformer_output
            )
            # logits = self.gpt2.forward(
            #     input_ids = input_ids,
            #     attention_mask = attention_mask
            # ).logits

        # If targets provided, compute loss
        if targets is not None:
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

            # Binary prediction eos yes/no
            if self.config["use_pad_head"]:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    masked_logits.flatten(),      # [B,]
                    torch.eq(masked_targets, self.config["pad_token_id"]).to(torch.float),    # [B,] same shape
                )

            # Next token prediction loss
            else:
                loss = torch.nn.functional.cross_entropy(
                    masked_logits,      # [B, vocab_size]   logits
                    masked_targets      # [B,]   target indices
                )

        # Otherwise, return None
        else:
            loss = None

        return logits, loss

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        This method processes a batch of data, performs a forward pass through the model,
        calculates the loss, logs various training metrics, and returns the loss.
        It handles different attention masking strategies (annealing for diffusion, causal for ARM)
        based on the configuration.
        Args:
            batch: A tuple containing input_ids, targets, and input_mask tensors.
            batch_idx: The index of the current batch.
        Returns:
            torch.Tensor: The calculated loss for the training step.
        """
        
        # Read in batch
        input_ids, targets, input_mask = batch        

        # Get shapes
        B, context_length = input_ids.shape
        
        # Diffusion --> anneal attention mask & mask input tokens
        if self.config["pipeline"] == "diffusion":
            # Mask input ids at specified positions
            input_ids[input_mask] = self.config["mask_id"]
            
            # Current annealing step (% of upper tril to be unmasked)
            if self.global_step < self.config["attn_annealing_steps"]:
                p = self.annealing_schedule[self.global_step]
            else:
                p = 1.0
            attention_mask = get_annealing_mask(context_length, B, p)
        
        # Arm --> get causal mask
        else:
            attention_mask = get_causal_mask(context_length, B)

        # Forward pass
        logits, loss = self.forward(
            input_ids = input_ids,
            targets = targets,
            input_mask = input_mask,
            attention_mask = attention_mask
        )
        
        # ============ Logging ========================
        if self.config["use_pad_head"]:
            
            # Get predictions and targets (1 if pad, 0 otherwise)
            flat_preds = torch.nn.functional.sigmoid(logits).round().view(-1)
            flat_targets = torch.eq(targets, self.config["pad_token_id"]).view(-1)
                
        else:
            # Get predictions from logits (1 if pad, 0 otherwise)
            flat_preds = torch.eq(logits.argmax(dim=-1), self.config["pad_token_id"]).view(-1)
            flat_targets = torch.eq(targets, self.config["pad_token_id"]).view(-1)

        # Compute metrics
        acc, rec, prec, f1 = compute_binary_metrics(flat_preds, flat_targets)

        # Save metrics 
        metrics = {
            "train/loss": loss,
            "train/learning_rate": self.optimizers().param_groups[0]['lr'],
            "train/masked_inputs_perc": input_mask.sum().item() / input_mask.numel(),
            "train/accuracy": acc,
            "train/recall": rec,
            "train/precision": prec,
            "train/f1": f1,
        }

        # Log all metrics
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        This method processes a batch of validation data. It handles attention mask
        creation based on the configured pipeline (diffusion or standard causal).
        For the diffusion pipeline, it masks input tokens according to `input_mask`
        and uses a fully visible attention mask (annealing factor 1.0).
        For other pipelines, it uses a standard causal attention mask.
        It then performs a forward pass to compute the loss and logits.
        The validation loss and the entropy of the model's predictions
        (as a percentage of the maximum possible entropy) are logged.
        Args:
            batch: A tuple containing input_ids, targets, and input_mask.
            batch_idx: The index of the current batch.
        Returns:
            The computed loss for the batch.
        """
        
        # Read in batch
        input_ids, targets, input_mask = batch
        assert input_ids.dtype == torch.int64

        # Get shapes
        B, context_length = input_ids.shape
        
        # Diffusion --> anneal attention mask & mask input tokens
        if self.config["pipeline"] == "diffusion":
            # Mask input ids at specified positions
            input_ids[input_mask] = self.config["mask_id"]
            # can see everything at validation
            attention_mask = get_annealing_mask(context_length, B, 1.0)
        else:
            attention_mask = get_causal_mask(context_length, B)
        
        # Forward pass
        logits, loss = self.forward(
            input_ids,
            targets,
            input_mask,
            attention_mask
        )
        
        # ============ Logging ========================
        if self.config["use_pad_head"]:
            
            # Get predictions and targets (1 if pad, 0 otherwise)
            flat_preds = torch.nn.functional.sigmoid(logits).round().view(-1)
            flat_targets = torch.eq(targets, self.config["pad_token_id"]).view(-1)
                
        else:
            # Get predictions from logits (1 if pad, 0 otherwise)
            argmax = logits.argmax(dim=-1)
            flat_preds = torch.eq(argmax, self.config["pad_token_id"]).view(-1)
            flat_targets = torch.eq(targets, self.config["pad_token_id"]).view(-1)

            # Compute entropy in the prediction (if 0, flat prediction)
            preds_ids = argmax.flatten().cpu().numpy()[input_mask.flatten().cpu().numpy()]   # only at masked positions
            _, counts = np.unique(
                preds_ids,
                return_counts = True
            )
            probs = counts / len(preds_ids)
            entropy = - np.sum(probs * np.log2(probs))
            max_entropy = np.log2(len(preds_ids))
            self.log(
                # Ratio of entropy in the prediction to max entroy
                # (proxy for how disperse, if close to 0 -> flat predictions)
                "valid/predictions_entropy_perc",
                entropy / max_entropy,
                on_epoch=True,
                on_step=False,
                prog_bar=True
            )

        # Compute metrics
        acc, rec, prec, f1 = compute_binary_metrics(flat_preds, flat_targets)

        # Save metrics 
        metrics = {
            "valid/loss": loss,
            "valid/learning_rate": self.optimizers().param_groups[0]['lr'],
            "valid/masked_inputs_perc": input_mask.sum().item() / input_mask.numel(),
            "valid/accuracy": acc,
            "valid/recall": rec,
            "valid/precision": prec,
            "valid/f1": f1,
        }
        
        # Log all metrics
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the model.

        This method sets up the AdamW optimizer and a OneCycleLR learning rate
        scheduler. It follows the common practice of applying weight decay only
        to parameters with 2 or more dimensions (typically weight matrices and
        embeddings), while excluding biases and Layer Normalization parameters
        (which have fewer dimensions) from weight decay. This separation is
        achieved by creating two parameter groups.

        Layer normalization parameters and biases already have a limited number of
        degrees of freedom, so they don't need regularization as much as weight
        matrices do. This approach has been shown empirically to lead to better
        convergence and model performance.

        The learning rate starts low, increases linearly during the warm-up phase,
        and then decreases following a cosine annealing schedule. Hyperparameters
        for the optimizer (learning rate, betas, weight decay) and the scheduler
        (warm-up percentage, division factors) are sourced from the model's
        configuration (`self.config`).

        Returns:
            dict: A dictionary containing the configured optimizer and
                  learning rate scheduler, formatted for use with PyTorch Lightning.
                  The dictionary includes the optimizer instance under the key
                  "optimizer" and the scheduler configuration under "lr_scheduler".
        """

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
    def generate(
        self,
        input_ids: torch.Tensor,
        pipeline: str,              # "diffusion" / "arm"
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        do_sample: bool = None,
        repetition_penalty: float = None,
        denoising_strategy: str = None,    # "random" / "entropy"
        diffusion_steps: int = None
    )->list[torch.Tensor]:
        """
        Samples from the model according to the specified pipeline. 
        Always returns a list of tensors.
        When pipeline is diffusion, this list has length equal to the
        number of diffusion steps, and each element is an intermediate 
        stage of the diffusion process.
        When pipeline is arm, the list has length 1.
        """
        if pipeline == "arm":
            assert do_sample is not None
            assert repetition_penalty is not None
            genconfig = GenerationConfig(
                max_new_tokens = max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            
            out = self.gpt2.generate(
                inputs=input_ids,
                generation_config=genconfig,
                do_sample=do_sample,
                repetition_penalty= repetition_penalty,
            )
            return [out]
        
        elif pipeline == "diffusion":
            assert diffusion_steps is not None
            assert denoising_strategy is not None
            xs = self.generate_diffusion(
                input_ids,
                max_new_tokens = max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                denoising_strategy=denoising_strategy,
                diffusion_steps = diffusion_steps
            )
            return xs
        
        else:
            print("[!] Pipeline not implemented, check config file.")
            exit()
        
    def generate_diffusion(
        self,
        input_ids,
        max_new_tokens,
        temperature,
        top_k,
        denoising_strategy:str,      # "random", "entropy"
        diffusion_steps: int,
    ):
        """
        Generate text using diffusion process.
        The total number T of diffusion steps is given.
        At each step, un-mask  1/T % of tokens. 
        The tokens to be un-masked are the ones with highest confidence (lowest entropy).
        Conversely, in the DiffuGPT paper they sample 1/T% tokens at random.
        """

        assert isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2
        assert temperature > 0
        #assert input_ids[0,0] == 50256  # <|endoftext|> token

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
        
        # Number of tokens to be denoised at each step
        n_tokens_per_step = math.ceil(1 / diffusion_steps * max_new_tokens)
        
        # Full attention mask for inference
        attention_mask = get_annealing_mask(seq_len, B, 1.0).to(x.device)
        
        # To store intermediate steps
        xs = [x.clone()]

        # Gradually unmask tokens
        for step in range(diffusion_steps, 0, -1):
            
            # Get masked positions
            mask = (x == self.config["mask_id"])

            # Exit when generation is completed
            if not torch.any(mask):
                break
            
            # Forward pass to get predictions
            with torch.no_grad():
                logits, _ = self.forward(
                    input_ids=x,
                    targets=None,
                    input_mask=None,
                    attention_mask=attention_mask
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
            # [0, 1, 2, 3, ..., n] -> [0, 0, 1, 2, ..., n-1]
            next_tokens = torch.cat(
                [next_tokens[:,0:1], next_tokens[:, :-1]],
                 dim=1
            )

            # Get indices of tokens to be denoised
            if denoising_strategy == "random":
                # Choose n_tokens_per_step at random among the masked ones
                step_mask = torch.zeros_like(mask).to(torch.bool)
                for b in range(step_mask.shape[0]):
                    nonzero = mask[b].nonzero().flatten().cpu().numpy()
                    np.random.shuffle(nonzero)
                    idx = nonzero[:n_tokens_per_step].tolist()
                    step_mask[b, idx] = True

            elif denoising_strategy == "entropy":
                # Compute entropy for each position (along the vocabulary)
                entropy = -torch.sum(
                    probs * torch.log2(probs + 1e10),   # avoid log(0) when top k is set
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

                # Get the most confident predictions (lowest entropy)
                _, idx_to_denoise = torch.topk(
                    entropy,
                    k=n_tokens_per_step,
                    dim=-1,
                    largest=False
                )

                # Transform indices to mask
                step_mask = torch.zeros_like(mask)
                step_mask.scatter_(1, idx_to_denoise, True)
            else:
                print("Denoising strategy must be one of [\"random\", \"entropy\"].")
                exit()

            # Only predict masked tokens with highest confidence
            mask = mask & step_mask
            
            # Update the tokens that were masked
            x = torch.where(mask, next_tokens, x)

            # Keep track of diffusion process
            xs.append(x.clone())
            
        return xs

    def eval_forward(
        self,
        input_ids,
        src_mask,
    ):
        """
        Evaluate a sentence using the model. In this case the originally masked tokens are at the end of the sentence.
        Returns perplexity (lower is better)
        """
        assert isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2
        #assert input_ids[0,0] == 50256  # <|endoftext|> token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval()
        input_ids = input_ids.to(device)
        # Get dimensions
        B = input_ids.shape[0]
        assert(B == 1) # just for now
        seq_len = input_ids.shape[1]
        
        # Start with fully masked sequence for the new tokens
        max_new_tokens = src_mask.shape[1] - torch.sum(src_mask[0]).item()
        x = torch.full((B, seq_len), self.config["mask_id"], dtype=torch.long, device=device)
        
        # (id, ..., id, msk, ..., msk)
        # [B, seq_len]
        x = torch.where(src_mask == 1, input_ids, x)
        
        # Number of diffusion steps
        num_steps = self.config.get("diffusion_steps", max(64, max_new_tokens // 4)) 
        
        # Number of tokens to be denoised at each step
        n_tokens_per_step = math.ceil(1 / num_steps * max_new_tokens)
        
        # Full attention mask for inference
        attention_mask = get_annealing_mask(seq_len, B, 1.0).to(x.device)
        
        # To store intermediate steps
        xs = [x.clone()]
        logit_score = 0.0

        accumulate_probas = torch.full((B, seq_len), -1.0, device=device)

        # Gradually unmask tokens
        for step in range(num_steps, 0, -1):
            
            # Get masked positions
            mask = (x == self.config["mask_id"])

            # Exit when generation is completed
            if not torch.any(mask):
                break
            
            # Forward pass to get predictions
            with torch.no_grad():
                logits, _ = self.forward(
                    input_ids=x,
                    targets=None,
                    input_mask=None,
                    attention_mask=attention_mask
                )
            
            # da capire se possono andare sotto zero
            positional_probas = torch.full((B, seq_len), -1.0, device=device)
            probas = torch.softmax(logits, dim=-1)
            for i in range(seq_len):
                if mask[0][i]:
                    positional_probas[0][i] = (probas[:, i, input_ids[:, i]].item())
            
            # print("positional_logits:", positional_logits)
            


            # Get the most confident predictions (lowest entropy)
            probas_to_denoise, idx_to_denoise = torch.topk(
                positional_probas,
                k=n_tokens_per_step,
                dim=-1,
                largest=True
            )
            logit_score += torch.sum(probas_to_denoise)
            
            # Add probas_to_denoise to accumulate_probas
            for i, idx in enumerate(idx_to_denoise[0]):
                accumulate_probas[0, idx] = probas_to_denoise[0, i]
            

            # Transform indices to mask
            step_mask = torch.zeros_like(mask)
            step_mask.scatter_(1, idx_to_denoise, True)

            # Only predict masked tokens with highest confidence
            mask = mask & step_mask
            
            # Update the tokens that were masked
            x = torch.where(mask, input_ids, x)
            # print(f"x[{step}]:", x)
            # Keep track of diffusion process
            xs.append(x.clone())

        # Compute perplexity on accumulate_probas, considering only nonnegative values
        valid_probas = accumulate_probas[0][accumulate_probas[0] >= 0]
        assert(valid_probas.shape[0] == max_new_tokens)
        perplexity = torch.exp(-torch.mean(torch.log(valid_probas)))
        return perplexity.item()

    def generate_infilling(
        self,
        input_ids,
        src_mask,
        temperature,
        top_k
    ):
        """
        Generate text using diffusion process.
        It fills in the masked tokens in the input sequence.
        """

        assert isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2
        #assert input_ids[0,0] == 50256  # <|endoftext|> token
        device = 'cuda'
        self.eval()
        input_ids = input_ids.to(device)
        # Get dimensions
        B = input_ids.shape[0]
        assert(B == 1) # just for now
        seq_len = input_ids.shape[1]
        
        # Start with fully masked sequence for the new tokens
        max_new_tokens = src_mask.shape[1] - torch.sum(src_mask[0]).item()
        x = torch.full((B, seq_len), self.config["mask_id"], dtype=torch.long, device=device)
        
        # (id, ..., id, msk, ..., msk)
        # [B, seq_len]
        x = torch.where(src_mask == 1, input_ids, x)
        
        # Number of diffusion steps
        num_steps = self.config.get("diffusion_steps", max(64, max_new_tokens // 4)) 
        
        # Number of tokens to be denoised at each step
        n_tokens_per_step = math.ceil(1 / num_steps * max_new_tokens)
        
        # Full attention mask for inference
        attention_mask = get_annealing_mask(seq_len, B, 1.0).to(x.device)
        
        # To store intermediate steps
        xs = [x.clone()]


        # Gradually unmask tokens
        for step in range(num_steps, 0, -1):
            
            # Get masked positions
            mask = (x == self.config["mask_id"])

            # Exit when generation is completed
            if not torch.any(mask):
                break
            
            # Forward pass to get predictions
            with torch.no_grad():
                logits, _ = self.forward(
                    input_ids=x,
                    targets=None,
                    input_mask=None,
                    attention_mask=attention_mask
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
            # [0, 1, 2, 3, ..., n] -> [0, 0, 1, 2, ..., n-1]
            next_tokens = torch.cat(
                [next_tokens[:,0:1], next_tokens[:, :-1]],
                 dim=1
            )
            
            denoising_strategy = self.config.get("denoising_strategy", "random")
            # Get indices of tokens to be denoised
            if denoising_strategy == "random":
                # Choose n_tokens_per_step at random among the masked ones
                step_mask = torch.zeros_like(mask).to(torch.bool)
                for b in range(step_mask.shape[0]):
                    nonzero = mask[b].nonzero().flatten().cpu().numpy()
                    np.random.shuffle(nonzero)
                    idx = nonzero[:n_tokens_per_step].tolist()
                    step_mask[b, idx] = True

            elif denoising_strategy == "entropy":
                # Compute entropy for each position (along the vocabulary)
                entropy = -torch.sum(
                    probs * torch.log2(probs + 1e10),   # avoid log(0) when top k is set
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

                # Get the most confident predictions (lowest entropy)
                _, idx_to_denoise = torch.topk(
                    entropy,
                    k=n_tokens_per_step,
                    dim=-1,
                    largest=False
                )

                # Transform indices to mask
                step_mask = torch.zeros_like(mask)
                step_mask.scatter_(1, idx_to_denoise, True)
            else:
                print("Denoising strategy must be one of [\"random\", \"entropy\"].")
                exit()

            # Only predict masked tokens with highest confidence
            mask = mask & step_mask
            
            # Update the tokens that were masked
            x = torch.where(mask, next_tokens, x)

            # Keep track of diffusion process
            xs.append(x.clone())
            
        return xs[-1]

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
            attention_mask = get_annealing_mask(seq_len, B, 1.0).to(self.device)  # Full attention for testing
            logits, _ = self.forward(input_ids, input_ids, mask, attention_mask)
            
            # Get prediction probabilities for masked positions
            probs = torch.softmax(logits[0, prompt_len:], dim=-1)
            
            # Get probability of EOS token at each position
            eos_probs = probs[:, eos_token_id].cpu().numpy()
            
            # Check if EOS has highest probability at any position
            predicted_tokens = torch.argmax(logits[0, prompt_len:], dim=-1).cpu().numpy()
            has_eos = eos_token_id in predicted_tokens
            # eos_pos = np.argmax(eos_probs) if has_eos else -1
            
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
    
    @classmethod
    def from_pretrained(self, path_to_ckpt):
        """
        Loads a model from a .ckpt file.
        It needs the corresponding config.json to live in the same folder.
        """
        # Check the config.json exists in the parent dir
        config_path = os.path.join(
            os.path.dirname(path_to_ckpt),
            "config.json"
        )
        assert os.path.exists(config_path)
        
        # Load the model
        # TODO: load weights manually 
        # (to avoid initializing diffugpt from pretrained diffugpt)
        model = GPT2.load_from_checkpoint(
            checkpoint_path = path_to_ckpt,
            config_path = config_path
        )
        print(f"Successfully loaded weights from {path_to_ckpt}")
        return model