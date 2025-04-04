import sys
import json
import wandb
import lightning as pl
from model import GPT2
from datamodule import MemmapDataModule
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar

if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    print("No path/to/config.json provided, defaulting to \'./config.json\'")
    CONFIG_PATH = './config.json'

# Configuration file
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Setup logging
if config["wandb"]:
    wandb.login()
    run = wandb.init(
        project=config["project_name"],
        config=config,
        name=config["run_name"] if config["run_name"] else None,
    )
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/bce", summary="min", step_metric="epoch")
    logger = WandbLogger(project=config["project_name"])
else:
    logger = CSVLogger(save_dir=".")

# Load the datamodule
datamodule = MemmapDataModule(CONFIG_PATH)

# Instantiate a model
model = GPT2(CONFIG_PATH)

# progress_bar = TQDMProgressBar(
#     refresh_rate=len(datamodule.train_dataloader()) // 4,  # print progress bar every 25% of train epoch
#     leave=True,  # leave the last one at epoch end
# )
# Init the trainer
trainer = pl.Trainer(
    max_epochs=config["n_epochs"],
    # max_steps=config['n_steps'],
    accelerator="auto",                     # recognizes device
    devices="auto",                         # how many devices to use
    precision='16-mixed',                          # to use amp 16
    logger=logger,
    log_every_n_steps=1,
    check_val_every_n_epoch=1,              # run valid loop every 1 train loop
    enable_checkpointing=False,             # if true, saves the most recent model after each epoch TODO: personalize checkpointing
    # callbacks=[progress_bar],
    enable_progress_bar=True,
    # gradient_clip_val=1,
    # accumulate_grad_batches=config['accumulate_grad'],
)

# Train
trainer.fit(model, datamodule)
