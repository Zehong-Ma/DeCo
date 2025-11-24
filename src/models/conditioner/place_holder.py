import torch
from src.models.conditioner.base import BaseConditioner

class PlaceHolderConditioner(BaseConditioner):
    def __init__(self, null_class=1000):
        super().__init__()
        self.null_condition = null_class

    def _impl_condition(self, y, metadata):
        y = torch.randint(0, self.null_condition, (len(y),)).cuda()
        return y

    def _impl_uncondition(self, y, metadata):
        return torch.full((len(y),), self.null_condition, dtype=torch.long).cuda()