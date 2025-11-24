import torch
import logging


class BaseAE(torch.nn.Module):
    def __init__(self, scale=1.0, shift=0.0):
        super().__init__()
        self.scale = scale
        self.shift = shift

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def encode(self, x):
        return self._impl_encode(x) #.to(torch.bfloat16)

    # @torch.autocast("cuda", dtype=torch.bfloat16)
    def decode(self, x):
        return self._impl_decode(x) #.to(torch.bfloat16)

    def _impl_encode(self, x):
        raise NotImplementedError

    def _impl_decode(self, x):
        raise NotImplementedError

def uint82fp(x):
    x = x.to(torch.float32)
    x = (x - 127.5) / 127.5
    return x

def fp2uint8(x):
    x = torch.clip_((x + 1) * 127.5 + 0.5, 0, 255).to(torch.uint8)
    return x

