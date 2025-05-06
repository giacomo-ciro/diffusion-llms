import torch
import lightning as pl

class LladaBackbone(pl.LightningModule):
    """
    Wraps the Llada diffusion model to get the logits from the transfromer backbone.
    """
    def __init__(self):
        super().__init__()
        
        # The transformer model
        self.transformer = torch.nn.Identity()

        # The head
        self.head = torch.nn.Identity

        # The loss
        self.loss = 0

    def forward(self, input_ids, target):
        # get
        return

    def training_step(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        return

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        return