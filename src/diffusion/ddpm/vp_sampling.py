import torch

from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *
from typing import Callable

def ode_step_fn(x, eps, beta, sigma, dt):
    return x + (-0.5*beta*x + 0.5*eps*beta/sigma)*dt

def sde_step_fn(x, eps, beta, sigma, dt):
    return x + (-0.5*beta*x + eps*beta/sigma)*dt + torch.sqrt(dt.abs()*beta)*torch.randn_like(x)

import logging
logger = logging.getLogger(__name__)

class VPEulerSampler(BaseSampler):
    def __init__(
            self,
            train_max_t=1000,
            guidance_fn: Callable = None,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.guidance_fn = guidance_fn
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.train_max_t = train_max_t

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps
        assert self.last_step > 0.0
        assert self.scheduler is not None

    def _impl_sampling(self, net, noise, condition, uncondition):
        batch_size = noise.shape[0]
        steps = torch.linspace(1.0, self.last_step, self.num_steps, device=noise.device)
        steps = torch.cat([steps, torch.tensor([0.0], device=noise.device)], dim=0)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise, ]
        eps_trajs = []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            beta = self.scheduler.beta(t_cur)
            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out = net(cfg_x, cfg_t*self.train_max_t, cfg_condition)
            eps = self.guidance_fn(out, self.guidance)
            if i < self.num_steps -1 :
                x0 = self.last_step_fn(x, eps, beta, sigma, -t_cur[0])
                x = self.step_fn(x, eps, beta, sigma, dt)
            else:
                x = x0 = self.last_step_fn(x, eps, beta, sigma, -self.last_step)
            x_trajs.append(x)
            eps_trajs.append(eps)
        eps_trajs.append(torch.zeros_like(x))
        return x_trajs, eps_trajs