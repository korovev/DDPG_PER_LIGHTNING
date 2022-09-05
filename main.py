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
from datetime import datetime


class WarmStartFillBufferCallback(Callback):
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if WARM_POPULATE > 0:
            trainer.model.populate(WARM_POPULATE)
        else:
            pass


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int = 5000):
        super().__init__()
        self.every = every
        self.prev = None
        self.dirpath = "ckpt"

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs
    ) -> None:
        if pl_module.global_step % self.every == 0:
            current = (
                Path(self.dirpath)
                / f"{datetime_run_start}"
                / f"latest-{pl_module.global_step}.ckpt"
            )
            trainer.save_checkpoint(current)

            if self.prev != None:
                print(self.prev.unlink())

            self.prev = current


if __name__ == "__main__":

    ckpt_path = Path(
        "~/Documents/currently_active_works/advanced_topics_RL_Capobianco/source/project/DDPG_PER_LIGHTNING/ckpt"
    )
    datetime_run_start = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

    wandb_logger = WandbLogger(project="DDPG", log_model="all")

    if TRAIN:
        model = DDPG(wandb_logger)

        trainer = Trainer(
            accelerator="auto",
            max_epochs=TRAINER_MAX_EPOCHS,
            val_check_interval=VAL_CHECK_INTERVAL,
            default_root_dir=".",
            callbacks=[WarmStartFillBufferCallback(), PeriodicCheckpoint()],
            logger=wandb_logger,
        )

        trainer.fit(model)

    else:
        model = checkpoint_model = DDPG.load_from_checkpoint(
            "DDPG/g5xi4a4o/checkpoints/latest-90000.ckpt"
        )

        trainer = Trainer(
            accelerator="auto",
            default_root_dir=".",
            logger=wandb_logger,
        )

        train_dataloader = model.test_dataloader()
        trainer.test(model, dataloaders=train_dataloader)
