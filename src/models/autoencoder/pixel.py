import torch
from src.models.autoencoder.base import BaseAE

class PixelAE(BaseAE):
    def __init__(self, scale=1.0, shift=0.0):
        super().__init__(scale, shift)

    def _impl_encode(self, x):
        return x/self.scale+self.shift

    def _impl_decode(self, x):
        return (x-self.shift)*self.scale