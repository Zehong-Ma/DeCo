import torch
import lightning.pytorch.loggers.wandb as wandb

setattr(wandb, '_WANDB_AVAILABLE', True)
torch.set_float32_matmul_precision('medium')

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)

import os
os.environ["NCCL_DEBUG"] = "WARN"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

