from ddpg import DDPG
from simple_config import WARM_POPULATE, TRAINER_MAX_EPOCHS, VAL_CHECK_INTERVAL, TRAIN
from dataset import RLDataset

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pathlib import Path

import numpy as np


class WarmStartFillBufferCallback(Callback):
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if WARM_POPULATE > 0:
            trainer.model.populate(WARM_POPULATE)
        else:
            pass


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int = 10000):
        super().__init__()
        self.every = every

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs
    ):
        if pl_module.global_step % self.every == 0:
            current = Path(self.dirpath) / f"latest-{pl_module.global_step}.ckpt"
            prev = Path("chkpt") / f"latest-{pl_module.global_step - self.every}.ckpt"
            trainer.save_checkpoint(current)
            prev.unlink(missing_ok=True)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="DDPG", log_model="all")

    if TRAIN:
        model = DDPG(wandb_logger)
    else:
        model = checkpoint_model = DDPG.load_from_checkpoint(
            "DDPG/g5xi4a4o/checkpoints/latest-90000.ckpt"
        )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=TRAINER_MAX_EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
        default_root_dir=".",
        callbacks=[WarmStartFillBufferCallback(), PeriodicCheckpoint()],
        logger=wandb_logger,
    )

    if TRAIN:
        trainer.fit(model)
    else:
        train_dataloader = model.train_dataloader()
        trainer.test(model, dataloaders=train_dataloader)
