import json


import torch
import lightning as pl
# from transformers import GPT2Config, GPT2Model
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

        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.weight_decay = config["weight_decay"]
        self.learning_rate = config["learning_rate"]
        self.betas = config["betas"]

        # Init the model
        if config["model_checkpoint"].startswith("gpt2"):
            chkp = config["model_checkpoint"]
            self.gpt2 = GPT.from_pretrained(
                model_type = chkp,
                override_args = None
            )
            print(f"Loaded GPT-2 from {chkp}.")
        else:
            gptconfig = GPTConfig(
                block_size=config["block_size"],
                n_embd=config["n_embd"],
                n_layer=config["n_layer"],
                n_head=config["n_head"],
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
        idx, targets = batch

        logits, loss = self.forward(idx, targets)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train/MLM", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("valid/MLM", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.gpt2.configure_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            betas=self.betas,
            device_type="cuda" if torch.cuda.is_available else "cpu"
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=2, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/mse",  # Metric to track
                "interval": "epoch",
                "frequency": 1,  # Check after each epoch
            },
        }
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if isinstance(idx, list):       # convert to tensor
            idx = torch.tensor(idx)
        if len(idx.shape) == 1:         # add batch dim if missing
            idx = idx.unsqueeze(0)
        return self.gpt2.generate(idx, max_new_tokens, temperature, top_k)
