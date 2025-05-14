import torch
from diffusion_llms.models.gpt2_diffusion import DiffuGPT
from diffusion_llms.utils import get_annealing_mask, get_causal_mask, compute_binary_metrics


class DiffuGPT2LengthHead(DiffuGPT):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Freeze all GPT-2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False
        # Unfreeze embeddings for pad/eos and mask tokens
        wte = self.gpt2.transformer.wte.weight
        pad_id = self.config["pad_token_id"]
        eos_id = self.config.get("eos_token_id", pad_id)
        if pad_id == eos_id:
            wte[-1].requires_grad = True
        else:
            wte[pad_id].requires_grad = True
            wte[eos_id].requires_grad = True
        mask_id = self.config["mask_id"]
        wte[mask_id].requires_grad = True
        # Initialize new head for pad prediction
        self.length_head = torch.nn.Linear(self.gpt2.lm_head.in_features, 1)
        # Loss function for binary classification
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, input_mask: torch.Tensor = None, attention_mask: torch.Tensor = None) -> tuple:
        """
        Forward pass for pad prediction head.
        Returns:
            logits: Tensor of shape [B, seq_len] (raw scores)
            loss: BCEWithLogitsLoss over masked positions (or None if targets is None)
        """
        transformer_output = self.gpt2.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.length_head(transformer_output).squeeze(-1)  # [B, seq_len]
        loss = None
        if targets is not None and input_mask is not None:
            logits_flat = logits.view(-1)
            targets_flat = torch.eq(targets.view(-1), self.config["pad_token_id"]).to(torch.float)
            mask_flat = input_mask.view(-1)
            masked_idx = torch.nonzero(mask_flat, as_tuple=True)[0]
            masked_logits = logits_flat[masked_idx]
            masked_targets = targets_flat[masked_idx]
            loss = self.loss_fn(masked_logits, masked_targets)
        return logits, loss

    def training_step(self, batch, batch_idx):
        input_ids, targets, input_mask = batch
        B, seq_len = input_ids.shape
        # Prepare attention mask
        if self.config["pipeline"] == "diffusion":
            input_ids[input_mask] = self.config["mask_id"]
            p = self.annealing_schedule[self.global_step] if self.global_step < self.config.get("attn_annealing_steps", 0) else 1.0
            attention_mask = get_annealing_mask(seq_len, B, p)
        else:
            attention_mask = get_causal_mask(seq_len, B)
        logits, loss = self.forward(input_ids, targets, input_mask, attention_mask)
        probs = torch.sigmoid(logits).view(-1)
        labels = torch.eq(targets.view(-1), self.config["pad_token_id"]).view(-1)
        acc, rec, prec, f1 = compute_binary_metrics(probs.round(), labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/accuracy", acc, on_step=True, prog_bar=True)
        self.log("train/recall", rec, on_step=True)
        self.log("train/precision", prec, on_step=True)
        self.log("train/f1", f1, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, targets, input_mask = batch
        B, seq_len = input_ids.shape
        # Prepare attention mask
        if self.config["pipeline"] == "diffusion":
            input_ids[input_mask] = self.config["mask_id"]
            attention_mask = get_annealing_mask(seq_len, B, 1.0)
        else:
            attention_mask = get_causal_mask(seq_len, B)
        logits, loss = self.forward(input_ids, targets, input_mask, attention_mask)
        probs = torch.sigmoid(logits).view(-1)
        labels = torch.eq(targets.view(-1), self.config["pad_token_id"]).view(-1)
        acc, rec, prec, f1 = compute_binary_metrics(probs.round(), labels)
        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid/recall", rec, on_step=False, on_epoch=True)
        self.log("valid/precision", prec, on_step=False, on_epoch=True)
        self.log("valid/f1", f1, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Given input_ids [B, seq_len], returns probability of pad at each position [B, seq_len].
        """
        self.eval()
        B, seq_len = input_ids.shape
        attention_mask = get_causal_mask(seq_len, B)
        transformer_output = self.gpt2.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.length_head(transformer_output).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs