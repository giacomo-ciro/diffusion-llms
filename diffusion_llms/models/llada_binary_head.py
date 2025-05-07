import torch
import lightning as pl
import torch.nn.functional as F
from llada import LladaBackbone
import numpy as np

class UsainLLada(LladaBackbone):
    """
    Wraps the Llada diffusion and adds a linear head to predict the length of the sequence.

    In generation, the model will predict the length of the sequence and then generate the sequence.
    Length, then condition the generation on the length and thus complete the sequence, by filling remaining tokens.
    """
    def __init__(self):
        super().__init__()
        self.eos_token_id = self.tokenizer.eos_token_id # tokenizer is inherited from LladaBackbone

        # freeze all transformer parameters
        for param in self.transformer.parameters():
            param.requires_grad = False

        # to be sure freeze lm head
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # add a linear head to predict wether a token is the end of the sequence or not
        self.is_eos_head = torch.nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, target=None, return_length=False):
        # Pass input through the transformer
        transformer_outputs = self.transformer(input_ids)
        hidden_states = transformer_outputs.last_hidden_state # get transformer output

        logits, eos_logits = None, None
        if return_length:
            # get the length of the sequence
            eos_logits = self.is_eos_head(hidden_states)
            return logits, eos_logits
        
        logits = self.lm_head(hidden_states)
        
        return logits, eos_logits
    

    def training_step(self, batch, batch_idx):
        X, y = batch  # X: input_ids, y: target_ids (shifted by one)
        # Forward pass to get EOS logits
        _, eos_logits = self.forward(X, return_length=True)  # eos_logits: [batch, seq_len, 1]
        eos_logits = eos_logits.squeeze(-1)  # [batch, seq_len]
        # Compute loss (y: [batch, seq_len])
        loss = self.compute_loss(eos_logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        _, eos_logits = self.forward(X, return_length=True)
        eos_logits = eos_logits.squeeze(-1)
        loss = self.compute_loss(eos_logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def generate(
        self,
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
            self.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        predict_length = self.predict_length(x, block_length=block_length)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        # predict the length of the sequence

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
                    logits = self.get_logits(x_)
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.get_logits(x)

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

    
    def compute_loss(self, y_pred, y):
        # compute loss

        y = (y == self.eos_token_id).float()
        return F.binary_cross_entropy_with_logits(y_pred, y)

        
    def predict_length(self, input_ids, block_length=128):
        """
        Predict the length of the sequence.
        Args:
            input_ids: A tensor of shape (1, L).
        Returns:
            A tensor of shape (1, L).
        """
        _, eos_logits = self.forward(input_ids, return_length=True)
        eos_logits = eos_logits.squeeze(-1)
        return torch.sigmoid(eos_logits)

