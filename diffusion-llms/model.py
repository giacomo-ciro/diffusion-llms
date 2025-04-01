import json

from gpt2 import GPT, GPTConfig
import lightning as pl
import torch

class GPT2(pl.LightningModule):
    """
    Wraps the GPT class from Andrej Karpathy NanoGPT, and adapts to Lightning framework.
    """

    def __init__(
        self,
    ):
        super().__init__()

        # Read config file
        with open("./config.json", "r") as f:
            config = json.load(f)
        
        # Set to the model
        for key, value in config.items():
            setattr(self, key, value)

        # Init the model
        self.gpt2 = GPT.from_pretrained(
            model_type = "gpt2",
            override_args = None
        )

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
