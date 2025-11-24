import torch
import torch.nn as nn
from typing import List

class BaseConditioner(nn.Module):
    def __init__(self):
        super(BaseConditioner, self).__init__()

    def _impl_condition(self, y, metadata)->torch.Tensor:
        raise NotImplementedError()

    def _impl_uncondition(self, y, metadata)->torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def __call__(self, y, metadata:dict={}):
        condition = self._impl_condition(y, metadata)
        uncondition = self._impl_uncondition(y, metadata)
        if condition.dtype in [torch.float64, torch.float32, torch.float16]:
            condition = condition.to(torch.bfloat16)
        if uncondition.dtype in [torch.float64,torch.float32, torch.float16]:
            uncondition = uncondition.to(torch.bfloat16)
        return condition, uncondition


class ComposeConditioner(BaseConditioner):
    def __init__(self, conditioners:List[BaseConditioner]):
        super().__init__()
        self.conditioners = conditioners

    def _impl_condition(self, y, metadata):
        condition = []
        for conditioner in self.conditioners:
            condition.append(conditioner._impl_condition(y, metadata))
        condition = torch.cat(condition, dim=1)
        return condition

    def _impl_uncondition(self, y, metadata):
        uncondition = []
        for conditioner in self.conditioners:
            uncondition.append(conditioner._impl_uncondition(y, metadata))
        uncondition = torch.cat(uncondition, dim=1)
        return uncondition