import json
import os

import torch
import lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from transformers import GPT2LMHeadModel, GPT2Config, GenerationConfig
from diffusion_llms.attention_patch import replace_attention_mask
from diffusion_llms.utils import get_causal_mask

replace_attention_mask()   

class GPT2(pl.LightningModule):
    """
    Standard ARM implementation using GPT2 as backbone.
    """

    def __init__(
        self,
        config_path,
    ):
        super().__init__()
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.init_gpt2(pretrained=False)

    def init_gpt2(
            self,
            pretrained: str=None,
        ):
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

    def forward(
        self,
        input_ids:torch.Tensor,
        targets:torch.Tensor,
        attention_mask:torch.Tensor
    ) -> tuple:
        return self.gpt2(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = targets
        )

    def training_step(self, batch, batch_idx):
        input_ids, targets, input_mask = batch        
        B, context_length = input_ids.shape
        attention_mask = get_causal_mask(context_length, B)
        _, loss = self.forward(
            input_ids = input_ids,
            targets = targets,
            attention_mask = attention_mask
        )
        metrics = {
            "train/loss": loss,
            "train/learning_rate": self.optimizers().param_groups[0]['lr'],
        }
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
        input_ids, targets, input_mask = batch
        B, context_length = input_ids.shape
        attention_mask = get_causal_mask(context_length, B)
        _, loss = self.forward(
            input_ids,
            targets,
            attention_mask
        )

        metrics = {
            "valid/loss": loss,
            "valid/learning_rate": self.optimizers().param_groups[0]['lr'],
        }
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
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=self.config["max_lr"],
            betas = self.config["betas"],
            fused=True
        )        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config["max_lr"],
            total_steps=self.config["n_steps"],
            pct_start=self.config["warmup_pct"],
            div_factor=self.config["div_factor"],
            final_div_factor=self.config["final_div_factor"],
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        do_sample: bool = None,
        repetition_penalty: float = None,
    )->list[torch.Tensor]:
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
    
    @classmethod
    def from_pretrained(self, path_to_ckpt):
        config_path = os.path.join(
            os.path.dirname(path_to_ckpt),
            "config.json"
        )
        assert os.path.exists(config_path)
        model = GPT2.load_from_checkpoint(
            checkpoint_path = path_to_ckpt,
            config_path = config_path
        )
        print(f"Successfully loaded weights from {path_to_ckpt}")
        return model