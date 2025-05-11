import json
import math

import numpy as np
import torch
import lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config


from diffusion_llms.attention_patch import replace_attention_mask
from diffusion_llms.utils import get_annealing_mask, compute_binary_metrics

# Modify the HF implementation so 
# the attention mask provided is actually used
# JACK: manually inspected the call tree and 
# printed all intermediate attention mask,
# confirmed that without using this patch 
# the provided attention mask is not considered
# --> calling this function is necessary
# to enable custom attention masks !
replace_attention_mask()

class DiffuGPT(pl.LightningModule):
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

        self.pad_id = self.config["pad_token_id"]
        self.eos_id = self.config["eos_token_id"]
        self.mask_id = self.config["mask_id"]
        #self.device = self.config["device"]
        self.random_mask_prob = self.config["random_mask_prob"]
        self.eos_window_max = self.config["eos_window_max"]
        self.window_annealing_steps = self.config["window_annealing_steps"]
        self.window_schedule = np.linspace(
            self.eos_window_max,
            0,
            self.window_annealing_steps
        )
        self.init_diffugpt()
        
        # For the annealing 
        # At each step i, the entries in the attention mask 
        # will be un-masked with probability annealing_schedule[i]
        # at the last annealing step, all entries will be unmasked
        self.att_annealing_steps = self.config.get("attn_annealing_steps", 0)
        if self.att_annealing_steps > 0:
            self.annealing_schedule = np.linspace(
                0,
                1,
                self.att_annealing_steps
            )
            
        # Extend the WTE and LM_HEAD if required
        if self.config["pad_token_id"] and self.config["pad_token_id"] == 50257:
            self.extend_vocab()
        

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
        batch_size, _, seq_len, _ = attention_mask.shape

        # 1D mask over valid (non-pad) tokens
        valid_tokens = (input_ids != self.pad_id)         # [B, T] bool

        # expand to [B,1,1,T] for “keys” dimension and [B,1,T,1] for “queries”
        key_mask   = valid_tokens[:, None, None, :]  # can be attended-to
        query_mask = valid_tokens[:, None, :, None]  # can attend

        # combine with your existing mask
        attention_mask = attention_mask & key_mask & query_mask
        # (still boolean; HF GPT2 will cast to float internally)

        # Forward pass
        logits = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

        loss = None


        # If targets provided, compute loss
        if targets is not None:
            B, L, V = logits.size()
            # Flatten logits and targets
            logits_flat = logits.view(-1, V)
            targets_flat = targets.view(-1)
            mask_flat = input_mask.view(-1)

            # Compute per-token negative log-likelihood
            log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
            nll_flat = -log_probs_flat[torch.arange(logits_flat.size(0), device=logits.device), targets_flat]
            # Zero out positions not masked
            nll_flat = nll_flat * mask_flat.to(nll_flat.dtype)
            nll = nll_flat.view(B, L)

            # Unweighted average loss per masked token
            loss = nll.sum() / mask_flat.sum() 

        return logits, loss

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step, with masking logic moved here.
        """
        # Unpack batch
        input_ids, targets = batch
        input_ids = input_ids.clone()  # avoid in-place on original
        targets = targets.to(self.device)
        input_ids = input_ids.to(self.device)

        B, context_length = input_ids.shape

        input_mask, f = self.mask_sequence(
            input_ids,
        )

        # apply mask token
        input_ids[input_mask] = self.mask_id

        # create attention mask (full visibility for diffusion)
        attention_mask = get_annealing_mask(context_length, B, 1.0).to(self.device)

        # forward pass
        logits, _ = self.forward(
            input_ids=input_ids,
            targets=None,
            input_mask=input_mask,
            attention_mask=attention_mask
        )

        loss = self.compute_loss(
            logits,
            input_ids,
            targets,
            torch.reciprocal(f),
            shift=True
        )


        # compute binary metrics on EOS-token prediction
        flat_preds   = (logits.argmax(dim=-1) == self.eos_id).view(-1)
        flat_targets = (targets               == self.eos_id).view(-1)
        acc, rec, prec, f1 = compute_binary_metrics(flat_preds, flat_targets)

        # log metrics
        metrics = {
            "train/loss":               loss,
            "train/learning_rate":      self.optimizers().param_groups[0]['lr'],
            "train/masked_inputs_perc": input_mask.sum().item() / input_mask.numel(),
            "train/accuracy":           acc,
            "train/recall":             rec,
            "train/precision":          prec,
            "train/f1":                 f1,
        }
        for name, val in metrics.items():
            self.log(name, val, on_step=True, on_epoch=False, prog_bar=True)

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
        input_ids, targets = batch
        assert input_ids.dtype == torch.int64

        # Get shapes
        B, context_length = input_ids.shape

        input_mask, _ = self.mask_sequence(
            input_ids
        )

        # Mask input ids at specified positions
        input_ids[input_mask] = self.config["mask_id"]
        # can see everything at validation
        attention_mask = get_annealing_mask(context_length, B, 1.0)
        
        # Forward pass
        logits, _ = self.forward(
            input_ids,
            None,
            input_mask,
            attention_mask
        )

        # Compute loss
        loss = self.compute_loss(
            logits=logits,
            x_t=input_ids,
            y=targets,
            dsigma=torch.reciprocal(input_mask.sum(dim=-1)),
            shift=True
        )
        
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
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        denoising_strategy: str = None,    # "random" / "entropy"
        diffusion_steps: int = None,
        var_len: bool = False,
    )->list[torch.Tensor]:
        """
        Samples from the model according using generate diffusion. 
        """
        assert diffusion_steps is not None
        assert denoising_strategy is not None
        xs = self.generate_diffusion(
            input_ids,
            max_new_tokens = max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            denoising_strategy=denoising_strategy,
            diffusion_steps = diffusion_steps,
            # var_len=var_len,
        )
        return xs
        
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

            # Identify masked tokens
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

            # check if we find pad or eos tokens in x, if we do, crop the sequence
            # to ensure batch works even if > 1, truncate up to highest length and pad 
            # to the max length

            # Keep track of diffusion process
            xs.append(x.clone())
            
        return xs


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

    def compute_loss(
        self,
        logits: torch.Tensor,       # (batch_size, seq_len, vocab_size)
        x_t: torch.Tensor,          # (batch_size, seq_len) — input ids at timestep t
        y: torch.Tensor,            # (batch_size, seq_len) — target ids
        dsigma: torch.Tensor,       # (batch_size,) — per-example weights
        shift: bool = False         # whether to do the “shift” (next-token) variant
    ) -> torch.Tensor:
        """
        Compute the weighted cross-entropy loss over masked tokens.

        Returns:
            final_loss: scalar — the dsigma-weighted average loss over all masked positions
        """
        batch_size, seq_len, vocab_size = logits.size()

        # build mask of where we're predicting
        loss_mask = x_t == self.mask_id             # (batch_size, seq_len)

        if shift:
            # drop last logit, first target, and first mask bit
            logits    = logits[:, :-1, :]               # (batch_size, seq_len-1, vocab_size)
            loss_mask = loss_mask[:, 1:]                # (batch_size, seq_len-1)
            y         = y[:, 1:]                        # (batch_size, seq_len-1)

        # compute per-token CE, keep all positions
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),            # (batch_size*(seq_len'), vocab_size)
            y.reshape(-1),                              # (batch_size*(seq_len'))
            reduction="none"
        ).float().reshape(batch_size, -1)               # (batch_size, seq_len')

        # zero out the loss where we didn't mask
        loss = loss.masked_fill(~loss_mask, 0.0)

        # final weighted average over all masked tokens
        final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()

        return final_loss
    
    def mask_sequence(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # ---- continuous-time sampling ----
        sampling_eps = 1e-3
        t = (1 - sampling_eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + sampling_eps

        sigma = t
        dsigma = torch.reciprocal(t)

        # maskable_mask: True for tokens that can be masked (not src)
        # src_mask: True for source tokens (not to be masked)
        # Here, assume src_mask is (input_ids != self.pad_id)
        src_mask = (input_ids != self.pad_id)
        x_t = self.transition(input_ids, sigma[:, None], maskable_mask=~src_mask)

        # loss_mask: where x_t is masked
        loss_mask = x_t == self.mask_id

        return loss_mask, dsigma
    
    def transition(self, x_0, sigma, maskable_mask):
        """
        Applies a transition operation to the input tensor `x_0` based on a probability 
        determined by `sigma` and a mask `maskable_mask`.

        Args:
            x_0 (torch.Tensor): The input tensor representing the current state.
            sigma (float or torch.Tensor): The probability of transitioning each element 
                in `x_0` to the mask token. Higher values increase the likelihood of 
                masking elements.
            maskable_mask (torch.Tensor): A boolean tensor of the same shape as `x_0` 
                indicating which elements are eligible for masking.

        Returns:
            torch.Tensor: A tensor `x_t` where elements in `x_0` have been replaced 
            with the mask token (`self.mask_id`) based on the transition probability 
            and the maskable mask.
        """
        # move_chance = 1 - (-sigma).exp()
        move_chance = sigma
        move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
        x_t = torch.where(move_indices, self.mask_id, x_0)
        return x_t