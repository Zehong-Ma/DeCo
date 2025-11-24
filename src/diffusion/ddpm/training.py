import torch
from typing import Callable
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler

def inverse_sigma(alpha, sigma):
    return 1/sigma**2
def snr(alpha, sigma):
    return alpha/sigma
def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, min=threshold)
def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, max=threshold)
def constant(alpha, sigma):
    return 1

class VPTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            train_max_t=1000,
            lognorm_t=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.train_max_t = train_max_t
    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        batch_size = x.shape[0]
        if self.lognorm_t:
            t = torch.randn(batch_size).to(x.device, x.dtype).sigmoid()
        else:
            t = torch.rand(batch_size).to(x.device, x.dtype)

        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        sigma = self.scheduler.sigma(t)
        x_t = alpha * x + noise * sigma
        out = net(x_t, t*self.train_max_t, y)
        weight = self.loss_weight_fn(alpha, sigma)
        loss = weight*(out - noise)**2

        out = dict(
            loss=loss.mean(),
        )
        return out


class DDPMTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn: Callable = constant,
            train_max_t=1000,
            lognorm_t=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.train_max_t = train_max_t

    def _impl_trainstep(self, net, ema_net, x, y, metadata=None):
        batch_size = x.shape[0]
        t = torch.randint(0, self.train_max_t, (batch_size,))
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        sigma = self.scheduler.sigma(t)
        x_t = alpha * x + noise * sigma
        out = net(x_t, t, y)
        weight = self.loss_weight_fn(alpha, sigma)
        loss = weight * (out - noise) ** 2

        out = dict(
            loss=loss.mean(),
        )
        return out