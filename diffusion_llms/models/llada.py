# Inspired by https://github.com/ML-GSAI/LLaDA/blob/main/generate.py. True LLaDA code

import torch
import lightning as pl
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np

class LladaBackbone(pl.LightningModule):
    """
    Wraps the Llada diffusion model to get the logits from the transfromer backbone.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
        base_model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
        # The transformer model
        self.transformer = base_model.model.transformer

        # The head
        self.lm_head = self.transformer.pop("ff_out")

        # The loss
        self.loss = 0
    
    def forward(self, input_ids, target=None):
        hidden_states = self.forward_hidden_repr(input_ids)
        logits = self.lm_head(hidden_states)
        return {"logits": logits}  # Return as dictionary with 'logits' key

    def training_step(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        return

    @torch.no_grad()
    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
    ):
        """
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
        """
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
            model.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (
                x[
                    :,
                    prompt.shape[1] + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length :,
                ]
                == mask_id
            )
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                logits_with_noise = self.add_gumbel_noise(
                    logits, temperature=temperature
                )
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # b, l
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = (
            torch.zeros(
                mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
            )
            + base
        )

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        return num_transfer_tokens

    def add_gumbel_noise(self, logits, temperature):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def compute_loss(self, logits, targets, input_mask):
        """
        Compute the loss.
        Args:
            logits: The output of the model.
            targets: The target labels.
            input_mask: The input mask.
        """
        # no chunking at all
        logits = logits.reshape(-1, logits.size(-1))  # [B * seq_len, vocab_size]
        targets = targets.reshape(-1)  # [B * seq_len]
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

    def get_logits(self, input_ids):
        """
        Get the logits from the model.
        Args:
            input_ids: The input ids.
        """
        # no chunking at all
        logits = self(input_ids)
        return logits
    

    def to(self, device):
        """
        Move the model to the specified device.
        Args:
            device: The device to move the model to.
        """
        super().to(device)
        self.transformer.to(device)
        self.lm_head.to(device)
        return self
    
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

    def get_eos_logits(self, input_text):
        """
        Get the logits for the EOS token directly.
        
        Args:
            input_text: The input text for which to get the EOS logits.
        
        Returns:
            eos_logits: Logits corresponding to the EOS token.
        """
        # Tokenize and prepare the input
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        # Full tensor with mask_id
        x = torch.full((1, 1024), 126336, dtype=torch.long).to(self.device)
        x[:, : input_ids.shape[1]] = input_ids.clone()

        # Forward pass through the model components
        hidden_states = self.transformer["wte"](x)
        hidden_states = self.transformer["emb_drop"](hidden_states)

        # Safely process through blocks
        for block in self.transformer["blocks"]:
            hidden_states = block(hidden_states)

        # Final layer norm and language model head
        hidden_states = self.transformer["ln_f"](hidden_states)
        logits = self.lm_head(hidden_states)

        # Extract EOS logits
        eos_logits = logits[:, :, self.tokenizer.eos_token_id].squeeze(0)  # (1024)
        
        return eos_logits