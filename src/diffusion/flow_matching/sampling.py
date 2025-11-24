import torch
import os

from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *

from typing import Callable


def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt

def sde_mean_step_fn(x, v, dt, s, w):
    return x + v * dt + s * w * dt

def sde_step_fn(x, v, dt, s, w):
    return x + v*dt + s * w* dt + torch.sqrt(2*w*dt)*torch.randn_like(x)

def sde_preserve_step_fn(x, v, dt, s, w):
    return x + v*dt + 0.5*s*w* dt + torch.sqrt(w*dt)*torch.randn_like(x)


import logging
logger = logging.getLogger(__name__)

class EulerSampler(BaseSampler):
    def __init__(
            self,
            w_scheduler: BaseScheduler = None,
            timeshift=1.0,
            guidance_interval_min: float = 0.0,
            guidance_interval_max: float = 1.0,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, noise, condition, uncondition):
        """
        sampling process of Euler sampler
        -
        """
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise,]
        v_trajs = []
        # print(steps)
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out = net(cfg_x, cfg_t, cfg_condition)
            # print(t_cur[0])
            if t_cur[0] > self.guidance_interval_min and t_cur[0] <= self.guidance_interval_max:
                guidance = self.guidance
                out = self.guidance_fn(out, guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            v = out
            s = ((1/dalpha_over_alpha)*v - x)/(sigma**2 - (1/dalpha_over_alpha)*dsigma_mul_sigma)
            if i < self.num_steps -1 :
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs

class EulerSamplerJiT(BaseSampler):
    def __init__(
            self,
            w_scheduler: BaseScheduler = None,
            timeshift=1.0,
            guidance_interval_min: float = 0.0,
            guidance_interval_max: float = 1.0,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, noise, condition, uncondition):
        """
        sampling process of Euler sampler
        -
        """
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        x_trajs = [noise,]
        v_trajs = []
        # print(steps)
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out = net(cfg_x, cfg_t, cfg_condition)
            out = (out - cfg_x)/(1.0-cfg_t.view(-1, 1, 1, 1)).clamp_min(5e-2) # pred v

            # print(t_cur[0])
            if t_cur[0] > self.guidance_interval_min and t_cur[0] <= self.guidance_interval_max:
                guidance = self.guidance
                out = self.guidance_fn(out, guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            v = out
            s = ((1/dalpha_over_alpha)*v - x)/(sigma**2 - (1/dalpha_over_alpha)*dsigma_mul_sigma)
            if i < self.num_steps -1 :
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs

class HeunSampler(BaseSampler):
    def __init__(
            self,
            scheduler: BaseScheduler = None,
            w_scheduler: BaseScheduler = None,
            exact_henu=False,
            guidance_interval_min: float = 0.0,
            guidance_interval_max: float = 1.0,
            timeshift=1.0,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.exact_henu = exact_henu
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max
        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps
            
        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        
        assert  self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, noise, condition, uncondition):
        """
        sampling process of Henu sampler
        -
        """
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        v_hat, s_hat = 0.0, 0.0
        x_trajs = [noise, ]
        v_trajs = []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            alpha_over_dalpha = 1/self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            t_hat = t_next
            t_hat = t_hat.repeat(batch_size)
            sigma_hat = self.scheduler.sigma(t_hat)
            alpha_over_dalpha_hat = 1 / self.scheduler.dalpha_over_alpha(t_hat)
            dsigma_mul_sigma_hat = self.scheduler.dsigma_mul_sigma(t_hat)

            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0
            if i == 0 or self.exact_henu:
                cfg_x = torch.cat([x, x], dim=0)
                cfg_t_cur = t_cur.repeat(2)
                out = net(cfg_x, cfg_t_cur, cfg_condition)
                # out = self.guidance_fn(out, self.guidance)
                if t_cur[0] > self.guidance_interval_min and t_cur[0] <= self.guidance_interval_max:
                    guidance = self.guidance
                    out = self.guidance_fn(out, guidance)
                else:
                    out = self.guidance_fn(out, 1.0)
                v = out
                s = ((alpha_over_dalpha)*v - x)/(sigma**2 - (alpha_over_dalpha)*dsigma_mul_sigma)
            else:
                v = v_hat
                s = s_hat
            x_hat = self.step_fn(x, v, dt, s=s, w=w)
            # henu correct
            if i < self.num_steps -1:
                cfg_x_hat = torch.cat([x_hat, x_hat], dim=0)
                cfg_t_hat = t_hat.repeat(2)
                out = net(cfg_x_hat, cfg_t_hat, cfg_condition)
                
                if t_cur[0] > self.guidance_interval_min and t_cur[0] <= self.guidance_interval_max:
                    guidance = self.guidance
                    out = self.guidance_fn(out, guidance)
                else:
                    out = self.guidance_fn(out, 1.0)
                
                v_hat = out
                s_hat = ((alpha_over_dalpha_hat)* v_hat - x_hat) / (sigma_hat ** 2 - (alpha_over_dalpha_hat) * dsigma_mul_sigma_hat)
                v = (v + v_hat) / 2
                s = (s + s_hat) / 2
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)
            x_trajs.append(x)
            v_trajs.append(v)
        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs