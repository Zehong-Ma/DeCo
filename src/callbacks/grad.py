import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
from torch.optim import Optimizer

class GradientMonitor(pl.Callback):
    """Logs the gradient norm"""

    def __init__(self, norm_type: int = 2):
        norm_type = float(norm_type)
        if norm_type <= 0:
            raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")
        self.norm_type = norm_type

    def on_before_optimizer_step(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            optimizer: Optimizer
    ) -> None:
        norms = grad_norm(pl_module, norm_type=self.norm_type)
        max_grad = torch.tensor([v for k, v in norms.items() if k != f"grad_{self.norm_type}_norm_total"]).max()
        pl_module.log_dict({'train/grad/max': max_grad, 'train/grad/total': norms[f"grad_{self.norm_type}_norm_total"]})