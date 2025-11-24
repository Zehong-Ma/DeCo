import math
import torch
from src.diffusion.base.scheduling import *


class LinearScheduler(BaseScheduler):
    def alpha(self, t) -> Tensor:
        return (t).view(-1, 1, 1, 1)
    def sigma(self, t) -> Tensor:
        return (1-t).view(-1, 1, 1, 1)
    def dalpha(self, t) -> Tensor:
        return torch.full_like(t, 1.0).view(-1, 1, 1, 1)
    def dsigma(self, t) -> Tensor:
        return torch.full_like(t, -1.0).view(-1, 1, 1, 1)

# SoTA for ImageNet!
class GVPScheduler(BaseScheduler):
    def alpha(self, t) -> Tensor:
        return torch.cos(t * (math.pi / 2)).view(-1, 1, 1, 1)
    def sigma(self, t) -> Tensor:
        return torch.sin(t * (math.pi / 2)).view(-1, 1, 1, 1)
    def dalpha(self, t) -> Tensor:
        return -torch.sin(t * (math.pi / 2)).view(-1, 1, 1, 1)
    def dsigma(self, t) -> Tensor:
        return torch.cos(t * (math.pi / 2)).view(-1, 1, 1, 1)
    def w(self, t):
        return torch.sin(t)**2

class ConstScheduler(BaseScheduler):
    def w(self, t):
        return torch.ones(1, 1, 1, 1).to(t.device, t.dtype)

from src.diffusion.ddpm.scheduling import VPScheduler
class VPBetaScheduler(VPScheduler):
    def w(self, t):
        return self.beta(t).view(-1, 1, 1, 1)



