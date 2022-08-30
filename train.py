from ddpg import DDPG

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

if __name__ == "__main__":
    model = DDPG()

    model.buffer.sample()

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=150,
        val_check_interval=50,
        logger=CSVLogger(save_dir="logs/"),
    )

    trainer.fit(model)
