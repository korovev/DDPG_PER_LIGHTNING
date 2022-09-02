from ddpg import DDPG
from simple_config import WARM_POPULATE, TRAINER_MAX_EPOCHS, VAL_CHECK_INTERVAL

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

import numpy as np


class WarmStartFillBufferCallback(Callback):
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if WARM_POPULATE > 0:
            trainer.model.populate(WARM_POPULATE)
        else:
            pass


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="DDPG", log_model="all")
    model = DDPG(wandb_logger)

    trainer = Trainer(
        accelerator="auto",
        # devices=1 if torch.cuda.is_available() else None,
        max_epochs=TRAINER_MAX_EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
        # logger=CSVLogger(save_dir="logs/"),
        callbacks=[WarmStartFillBufferCallback()],
        logger=wandb_logger,
    )

    trainer.fit(model)
