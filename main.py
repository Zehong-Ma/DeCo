import os
import torch
import time
from typing import Any, Union

from src.utils.patch_bugs import *
from lightning import Trainer, LightningModule

from src.lightning_data import DataModule
from src.lightning_model import LightningModel
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback

import logging
logger = logging.getLogger("lightning.pytorch")
# log_path = os.path.join( f"log.txt")
# logger.addHandler(logging.FileHandler(log_path))

class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )


class ReWriteRootDirCli(LightningCLI):
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config_trainer = self._get(self.config, "trainer", default={})

        # predict path & logger check
        if self.subcommand == "predict":
            config_trainer.logger = None

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help=("torch hub dir"),)
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help=("huggingface hub dir"),)

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)
        is_resume = self._get(self.config_init, "ckpt_path", default=None)
        if os.path.exists(default_root_dir) and "debug" not in default_root_dir:
            if os.listdir(default_root_dir) and self.subcommand != "predict" and not is_resume:
                raise FileExistsError(f"{default_root_dir} already exists")

        config_trainer.default_root_dir = default_root_dir
        trainer = super().instantiate_trainer(**kwargs)
        if trainer.is_global_zero:
            os.makedirs(default_root_dir, exist_ok=True)
        return trainer

    def instantiate_classes(self) -> None:
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")
        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)
        super().instantiate_classes()

if __name__ == "__main__":

    cli = ReWriteRootDirCli(LightningModel, DataModule,
                            auto_configure_optimizers=False,
                            save_config_callback=ReWriteRootSaveConfigCallback,
                            save_config_kwargs={"overwrite": True})