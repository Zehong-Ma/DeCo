import torch
from src.models.autoencoder.base import BaseAE

class LatentAE(BaseAE):
    def __init__(self, precompute=False, weight_path:str=None):
        super().__init__()
        self.precompute = precompute
        self.model = None
        self.weight_path = weight_path

        from diffusers.models import AutoencoderKL
        setattr(self, "model", AutoencoderKL.from_pretrained(self.weight_path))
        self.scaling_factor = self.model.config.scaling_factor

    def _impl_encode(self, x):
        assert self.model is not None
        if self.precompute:
            return x.mul_(self.scaling_factor)
        encodedx =  self.model.encode(x).latent_dist.sample().mul_(self.scaling_factor)
        return encodedx

    def _impl_decode(self, x):
        assert self.model is not None
        return self.model.decode(x.div_(self.scaling_factor)).sample