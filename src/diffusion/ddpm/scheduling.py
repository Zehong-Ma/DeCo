import math
import torch
from src.diffusion.base.scheduling import *


class DDPMScheduler(BaseScheduler):
    def __init__(
            self,
            beta_min=0.0001,
            beta_max=0.02,
            num_steps=1000,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_steps = num_steps

        self.betas_table = torch.linspace(self.beta_min, self.beta_max, self.num_steps, device="cuda")
        self.alphas_table = torch.cumprod(1-self.betas_table, dim=0)
        self.sigmas_table = 1-self.alphas_table


    def beta(self, t) -> Tensor:
        t = t.to(torch.long)
        return self.betas_table[t].view(-1, 1, 1, 1)

    def alpha(self, t) -> Tensor:
        t = t.to(torch.long)
        return self.alphas_table[t].view(-1, 1, 1, 1)**0.5

    def sigma(self, t) -> Tensor:
        t = t.to(torch.long)
        return self.sigmas_table[t].view(-1, 1, 1, 1)**0.5

    def dsigma(self, t) -> Tensor:
        raise NotImplementedError("wrong usage")

    def dalpha_over_alpha(self, t) ->Tensor:
        raise NotImplementedError("wrong usage")

    def dsigma_mul_sigma(self, t) ->Tensor:
        raise NotImplementedError("wrong usage")

    def dalpha(self, t) -> Tensor:
        raise NotImplementedError("wrong usage")

    def drift_coefficient(self, t):
        raise NotImplementedError("wrong usage")

    def diffuse_coefficient(self, t):
        raise NotImplementedError("wrong usage")

    def w(self, t):
        raise NotImplementedError("wrong usage")


class VPScheduler(BaseScheduler):
    def __init__(
            self,
            beta_min=0.1,
            beta_max=20,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_d = beta_max - beta_min
    def beta(self, t) -> Tensor:
        t = torch.clamp(t, min=1e-3, max=1)
        return (self.beta_min + (self.beta_d * t)).view(-1, 1, 1, 1)

    def sigma(self, t) -> Tensor:
        t = torch.clamp(t, min=1e-3, max=1)
        inter_beta:Tensor = 0.5*self.beta_d*t**2 + self.beta_min* t
        return (1-torch.exp_(-inter_beta)).sqrt().view(-1, 1, 1, 1)

    def dsigma(self, t) -> Tensor:
        raise NotImplementedError("wrong usage")

    def dalpha_over_alpha(self, t) ->Tensor:
        raise NotImplementedError("wrong usage")

    def dsigma_mul_sigma(self, t) ->Tensor:
        raise NotImplementedError("wrong usage")

    def dalpha(self, t) -> Tensor:
        raise NotImplementedError("wrong usage")

    def alpha(self, t) -> Tensor:
        t = torch.clamp(t, min=1e-3, max=1)
        inter_beta: Tensor = 0.5 * self.beta_d * t ** 2 + self.beta_min * t
        return torch.exp(-0.5*inter_beta).view(-1, 1, 1, 1)

    def drift_coefficient(self, t):
        raise NotImplementedError("wrong usage")

    def diffuse_coefficient(self, t):
       raise NotImplementedError("wrong usage")

    def w(self, t):
        return self.diffuse_coefficient(t)



