from datamodule import MemmapDataModule
from transformers import GPT2Tokenizer

from torch.utils.data import DataLoader
import sys

# Inizializza il tuo datamodule
dm = MemmapDataModule("./local_config.json")
print("loaded")
dm.prepare_data()
print("prepared")
dm.setup()
print("setup done")

# Ottieni il dataloader
tl = dm.train_dataloader()
tl.num_workers = 0
print("train loader")
# Prendi un batch
# senza mettere num_workers = 0 è molto più lento
batch = next(iter(tl))
print("done")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
token_ids = batch[0][0]
print(len(token_ids))
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
print(decoded_text)