import os.path
from typing import Optional, Dict, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


class CheckpointHook(ModelCheckpoint):
    """Save checkpoint with only the incremental part of the model"""
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.dirpath = trainer.default_root_dir
        self.exception_ckpt_path = os.path.join(self.dirpath, "on_exception.pt")
        pl_module.strict_loading = False

    def on_save_checkpoint(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            checkpoint: Dict[str, Any]
    ) -> None:
        del checkpoint["callbacks"]

    # def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
    #     if not "debug" in self.exception_ckpt_path:
    #         trainer.save_checkpoint(self.exception_ckpt_path)