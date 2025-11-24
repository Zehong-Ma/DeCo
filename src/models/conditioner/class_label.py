import torch
from src.models.conditioner.base import BaseConditioner

class LabelConditioner(BaseConditioner):
    def __init__(self, num_classes):
        super().__init__()
        self.null_condition = num_classes

    def _impl_condition(self, y, metadata):
        return torch.tensor(y).long().cuda()

    def _impl_uncondition(self, y, metadata):
        return torch.full((len(y),), self.null_condition, dtype=torch.long).cuda()