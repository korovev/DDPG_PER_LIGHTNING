from ddpg import DDPG

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule

import numpy as np


class WarmStartFillBufferCallback(Callback):
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        trainer.model.populate(1000)


if __name__ == "__main__":
    model = DDPG()

    trainer = Trainer(
        accelerator="auto",
        # devices=1 if torch.cuda.is_available() else None,
        max_epochs=150,
        val_check_interval=50,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[WarmStartFillBufferCallback()],
    )

    trainer.fit(model)
