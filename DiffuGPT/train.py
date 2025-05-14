import os
import time
import sys
import json
import lightning as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from diffusion_llms.datamodule import MemmapDataModule
from diffusion_llms.utils import check_config_validity
from diffusion_llms.models.gpt2_arm import GPT2
from diffusion_llms.models.gpt2_diffusion import DiffuGPT
from diffusion_llms.models.diffugpt2_length_head import DiffuGPT2LengthHead

def init_model(config, config_path):
    """
    Initialize the model based on the configuration.
    """
    if os.path.exists(config["init_from"]):
        return GPT2.from_pretrained(config["init_from"])

    if config["pipeline"] == "arm":
        return GPT2(config_path)
    if config["pipeline"] == "diffusion":
        return DiffuGPT(config_path)
    elif config["pipeline"] == "diffusion_length_head":
        return DiffuGPT2LengthHead(config_path)
    else:
        raise ValueError(f"Unknown pipeline: {config['pipeline']}")


def init_model_checkpoint(config):
    # Unique folder for this run
    dirpath = config["save_dir"] + time.strftime("ymd_%y%m%d_HMS_%H_%M_%S")
    
    # Create also the save_dir folder if not exist
    os.makedirs(dirpath)
    
    # Create checkpointer object
    checkpointer = ModelCheckpoint(
        dirpath=dirpath,  # Directory to save checkpoints
        filename='epoch_{epoch}_ce_{valid/loss:.2f}',            # Checkpoint filename format
        save_top_k=3,                                                          # Save the 3 best models
        monitor='valid/loss',                                                     # Metric to monitor
        mode='min',                                                             # Mode ('min' for loss, 'max' for accuracy)
        auto_insert_metric_name=False,
    )
    
    # Save model config
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump(config, f, indent = 2)

    return checkpointer

def setup_logging(config):
    """
    Setup logging based on the configuration.
    """
    if config["wandb"]:
        logger = WandbLogger(
            name=config["run_name"] if config["run_name"] else None,
            project=config["project_name"]
        )
        logger.experiment.config.update(config)
    else:
        logger = CSVLogger(save_dir=".")
    
    return logger


def main():
    # From the command line we can specify the config.file
    if len(sys.argv) == 2:
        CONFIG_PATH = sys.argv[1]
    else:
        print("No path/to/config.json provided, defaulting to \'./config.json\'")
        CONFIG_PATH = './config.json'

    # Configuration file
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Check validity of configuration
    check_config_validity(config)

    logger = setup_logging(config)

    # Load the datamodule
    datamodule = MemmapDataModule(CONFIG_PATH)

    model = init_model(config, CONFIG_PATH)

    # Checkpointers
    if config["enable_checkpointing"]:
        checkpointer = init_model_checkpoint(config)
    
    # Early Stopping
    early_stopping = EarlyStopping(
        monitor='valid/loss',           # Monitor validation cross-entropy loss
        patience=2,                     # Number of validation checks with no improvement after which training will stop
        min_delta=0.001,                # Minimum change in monitored value to qualify as improvement
        mode='min',                     # We want to minimize the loss
        verbose=True,                   # Print message when early stopping is triggered
        check_on_train_epoch_end=False  # Check at validation time, not training end
    )

    # Init the trainer
    trainer = pl.Trainer(
        max_epochs=config["n_epochs"] if config["n_epochs"] else None,
        max_steps=config['n_steps'],                # stops when one of the two is met
        accelerator="mps",                         # recognizes device
        devices="auto",                             # how many devices to use
        precision='16-mixed',                       # to use amp 16
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=config["val_check_interval"],     # after how many train batches to check val
        callbacks=[checkpointer, early_stopping] if config["enable_checkpointing"] else [early_stopping],
        enable_progress_bar=True,
        accumulate_grad_batches=config['accumulate_grad'],
        # gradient_clip_val=config["grad_clip"],      # Maximum norm of the gradients
        # gradient_clip_algorithm='norm',             # 'norm' or 'value'
        # enable_checkpointing=False,                 # if true, saves the most recent model after each epoch TODO: personalize checkpointing
    )

    # Train
    # Epoch 0:  43%|▍| 378/883 [...
    #                   ^^ number of forward calls (if accumulate_grad == 1, then it coincides with optimizer steps)
    trainer.fit(
        model,
        datamodule,
        ckpt_path=config["init_from"] if config["resume_training"] else None
    )

if __name__ == "__main__":
    main()