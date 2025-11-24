from typing import Dict, Any, Optional

import torch
import torch.nn as nn


import logging
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self,):
        super().__init__()

    def load(self, denoiser):
        if denoiser.weight_path:
            weight = torch.load(denoiser.weight_path, map_location=torch.device('cpu'))

            if denoiser.load_ema:
                prefix = "ema_denoiser."
            else:
                prefix = "denoiser."
            for k, v in denoiser.state_dict().items():
                try:
                    v.copy_(weight["state_dict"][prefix+k])
                except:
                    logger.warning(f"Failed to copy {prefix+k} to denoiser weight")
        return denoiser