import torch
from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *

from typing import Callable

import logging
logger = logging.getLogger(__name__)

class DDIMSampler(BaseSampler):
    def __init__(
            self,
            train_num_steps=1000,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.train_num_steps = train_num_steps
        assert self.scheduler is not None

    def _impl_sampling(self, net, noise, condition, uncondition):
        batch_size = noise.shape[0]
        steps = torch.linspace(0.0, self.train_num_steps-1, self.num_steps, device=noise.device)
        steps = torch.flip(steps, dims=[0])
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = x0 = noise
        x_trajs = [noise, ]
        v_trajs = []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            t_cur = t_cur.repeat(batch_size)
            t_next = t_next.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            alpha = self.scheduler.alpha(t_cur)
            sigma_next = self.scheduler.sigma(t_next)
            alpha_next = self.scheduler.alpha(t_next)
            cfg_x = torch.cat([x, x], dim=0)
            t = t_cur.repeat(2)
            out = net(cfg_x, t, cfg_condition)
            out = self.guidance_fn(out, self.guidance)
            x0 = (x - sigma * out) / alpha
            x = alpha_next * x0 + sigma_next * out
            x_trajs.append(x)
            v_trajs.append(out)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs