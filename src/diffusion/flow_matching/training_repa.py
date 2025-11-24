import torch
import copy
import timm
from torch.nn import Parameter

from src.utils.no_grad import no_grad
from typing import Callable, Iterator, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
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


def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)


class REPATrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            feat_loss_weight: float=0.5,
            lognorm_t=False,
            timeshift=1.0,
            encoder:nn.Module=None,
            align_layer=8,
            proj_denoiser_dim=256,
            proj_hidden_dim=256,
            proj_encoder_dim=256,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.align_layer = align_layer
        self.encoder = encoder
        no_grad(self.encoder)

        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raw_images = metadata["raw_image"]
        batch_size, c, height, width = x.shape
        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=x.device, dtype=torch.float32).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift) #.to(x.dtype)
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        x_t = alpha * x + noise * sigma
        v_t = dalpha * x + dsigma * noise

        src_feature = []
        def forward_hook(net, input, output):
            feature = output
            if isinstance(feature, tuple):
                feature = feature[0] # mmdit
            src_feature.append(feature)
        if getattr(net, "encoder", None) is not None:
            handle = net.encoder.blocks[self.align_layer - 1].register_forward_hook(forward_hook)
        else:
            handle = net.blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        out = net(x_t, t, y)
        src_feature = self.proj(src_feature[0])
        handle.remove()

        with torch.no_grad():
            dst_feature = self.encoder(raw_images)
        if dst_feature.shape[1] != src_feature.shape[1]:
            src_feature = src_feature[:, :dst_feature.shape[1]]


        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)
        cos_loss = 1 - cos_sim

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight*(out - v_t)**2
        out = dict(
            fm_loss=fm_loss.mean(),
            cos_loss=cos_loss.mean(),
            loss=fm_loss.mean() + self.feat_loss_weight*cos_loss.mean(),
        )

        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars)

