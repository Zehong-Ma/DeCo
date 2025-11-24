import time

import torch
import torch.nn as nn


class BaseTrainer(nn.Module):
    def __init__(self,
                 null_condition_p=0.1,
        ):
        super(BaseTrainer, self).__init__()
        self.null_condition_p = null_condition_p

    def preproprocess(self, x, condition, uncondition, metadata):
        bsz = x.shape[0]
        if self.null_condition_p > 0:
            mask = torch.rand((bsz), device=condition.device) < self.null_condition_p
            mask = mask.view(-1, *([1] * (len(condition.shape) - 1))).to(condition.dtype)
            condition = condition*(1-mask)  + uncondition*mask
        return x, condition, metadata

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raise NotImplementedError

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def __call__(self, net, ema_net, solver, x, condition, uncondition, metadata=None):
        x, condition, metadata = self.preproprocess(x, condition, uncondition, metadata)
        return self._impl_trainstep(net, ema_net, solver, x, condition, metadata)

