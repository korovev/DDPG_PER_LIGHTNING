from ddpg import DDPG

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

import numpy as np


class WarmStartFillBufferCallback(Callback):
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        trainer.model.populate(1000)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="DDPG", log_model="all")
    model = DDPG()

    trainer = Trainer(
        accelerator="auto",
        # devices=1 if torch.cuda.is_available() else None,
        max_epochs=150000,
        val_check_interval=100,
        # logger=CSVLogger(save_dir="logs/"),
        callbacks=[WarmStartFillBufferCallback()],
        logger=wandb_logger,
    )

    trainer.fit(model)
