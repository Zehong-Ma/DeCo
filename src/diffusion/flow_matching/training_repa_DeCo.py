import torch
import copy
import timm
from torch.nn import Parameter
from torch import nn
import torch.nn.functional as F

from src.utils.no_grad import no_grad
from typing import Callable, Iterator, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler

from torchvision.utils import save_image
import os

def inverse_sigma(alpha, sigma):
    return 1/sigma**2
def snr(alpha, sigma):
    return alpha/sigma
def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, min=threshold)
def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, max=threshold)
def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)


class REPATrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            feat_loss_weight: float=0.5,
            lognorm_t=False,
            timeshift=1.0,
            encoder:nn.Module=None,
            align_layer=8,
            proj_denoiser_dim=256,
            proj_hidden_dim=256,
            proj_encoder_dim=256,
            freq_loss_weight=1,
            freq_quality: int = 85,
            freq_mode: str = 'inv_gamma',
            freq_gamma: float = 1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.align_layer = align_layer
        self.freq_loss_weight = freq_loss_weight
        self.encoder = encoder
        no_grad(self.encoder)

        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )

        # DCT 配置
        self.block_size = 8
        self.register_buffer("dct_mat", self._create_dct_matrix(self.block_size))
        self.register_buffer("freq_w", self._build_freq_weight(
            quality=freq_quality, mode=freq_mode, gamma=freq_gamma
        ))

    # ===== DCT / freq utils =====
    def _create_dct_matrix(self, N: int):
        import math
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        C = torch.cos(math.pi * (2*n + 1) * k / (2.0 * N))
        alpha = torch.sqrt(torch.tensor(2.0) / N) * torch.ones(N)
        alpha[0] = math.sqrt(1.0 / N)
        C = alpha.unsqueeze(1) * C
        return C  # (N,N)

    def _rgb2ycbcr(self, x: torch.Tensor):
        """RGB -> YCbCr, x in [B,C,H,W]"""
        r = x[:,0:1,:,:]
        g = x[:,1:2,:,:]
        b = x[:,2:3,:,:]
        y  = 0.299*r + 0.587*g + 0.114*b
        cb = -0.168736*r - 0.331264*g + 0.5*b
        cr = 0.5*r - 0.418688*g -0.081312*b
        return torch.cat([y, cb, cr], dim=1)  # (B,3,H,W)

    @torch.compile()
    def _dct(self, x: torch.Tensor):
        """
        8x8 block DCT, 返回 (B,C,Bh,Bw,bs,bs)
        """
        bs = self.block_size
        B, C, H, W = x.shape
        pad_h = (-H) % bs
        pad_w = (-W) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0,pad_w,0,pad_h), mode="reflect")

        B, C, H2, W2 = x.shape
        Bh, Bw = H2//bs, W2//bs
        blocks = x.unfold(2, bs, bs).unfold(3, bs, bs)  # (B,C,Bh,Bw,bs,bs)
        blocks = blocks.contiguous().view(-1, bs, bs)

        Cmat = self.dct_mat.to(x.device, x.dtype)
        dct_flat = torch.matmul(Cmat.unsqueeze(0), blocks)
        dct_flat = torch.matmul(dct_flat, Cmat.t().unsqueeze(0))
        return dct_flat.view(B, C, Bh, Bw, bs, bs)

    def _build_freq_weight(self, quality=85, mode='inv_gamma', gamma=1.0):
        # JPEG luminance & chrominance base tables
        lum_q = torch.tensor([
            [16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]], dtype=torch.float32)

        chr_q = torch.tensor([
            [17,18,24,47,99,99,99,99],
            [18,21,26,66,99,99,99,99],
            [24,26,56,99,99,99,99,99],
            [47,66,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99]], dtype=torch.float32)

        def scale_q(base_q, quality):
            q = max(1, min(100, int(quality)))
            if q < 50:
                scale = 5000 / q
            else:
                scale = 200 - 2*q
            return torch.floor((base_q*scale + 50)/100).clamp(1,255)

        Q_y = scale_q(lum_q, quality)
        Q_cbcr = scale_q(chr_q, quality)

        def q_to_weight(Q):
            if mode == 'inv':
                w = 1.0 / Q
            elif mode == 'inv_gamma':
                w = (Q.mean()/Q) ** gamma
            else:
                raise ValueError("mode must be 'inv' or 'inv_gamma'")
            return w / w.mean()

        # Y -> luminance, Cb/Cr -> chrominance
        w_y = q_to_weight(Q_y)
        w_cb = q_to_weight(Q_cbcr)
        w_cr = q_to_weight(Q_cbcr)

        # stack成 (1,C,1,1,8,8)
        w = torch.stack([w_y, w_cb, w_cr], dim=0)
        return w.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1,3,1,1,8,8)
    
    # def save_tensor_as_images(self, tensor, save_dir="debug_output", prefix="img"):
    #     """
    #     将 [0,1] 值域的 BCHW 张量保存为图片
    #     Args:
    #         tensor (torch.Tensor): BCHW 格式，值域 [0,1]
    #         save_dir (str): 保存目录
    #         prefix (str): 文件名前缀
    #     """
    #     os.makedirs(save_dir, exist_ok=True)

    #     # 确保在 [0,1] 范围内
    #     # tensor = torch.clamp(tensor, 0, 1)
        
    #     B = tensor.size(0)
    #     for i in range(B):
    #         save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.png")
    #         save_image(tensor[i], save_path)
    #         print(f"✅ 已保存: {save_path}")
    
    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raw_images = metadata["raw_image"]
        batch_size, c, height, width = x.shape
        # self.save_tensor_as_images(raw_images)
        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=x.device, dtype=torch.float32).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift)
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        x_t = alpha * x + noise * sigma
        v_t = dalpha * x + dsigma * noise
        
        

        src_feature = []
        def forward_hook(net, input, output):
            feature = output
            if isinstance(feature, tuple):
                feature = feature[0] # mmdit
            src_feature.append(feature)
        if getattr(net, "encoder", None) is not None:
            handle = net.encoder.blocks[self.align_layer - 1].register_forward_hook(forward_hook)
        else:
            handle = net.blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        out = net(x_t, t, y)
        src_feature = self.proj(src_feature[0])
        handle.remove()

        with torch.no_grad():
            dst_feature = self.encoder(raw_images)

        if dst_feature.shape[1] != src_feature.shape[1]:
            src_feature = src_feature[:, :dst_feature.shape[1]]


        cos_sim = torch.nn.functional.cosine_similarity(src_feature, dst_feature, dim=-1)
        cos_loss = 1 - cos_sim

        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight*(out - v_t)**2

        v_t_freq = self._dct(self._rgb2ycbcr(v_t))
        out_freq = self._dct(self._rgb2ycbcr(out))
        fm_loss_freq = (self.freq_w*((out_freq - v_t_freq)**2)).mean()

        out = dict(
            fm_loss=fm_loss.mean(),
            fm_loss_freq=fm_loss_freq, 
            cos_loss=cos_loss.mean(),
            loss=fm_loss.mean()+self.freq_loss_weight*fm_loss_freq+ self.feat_loss_weight*cos_loss.mean(),
        )
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.proj.state_dict(
            destination=destination,
            prefix=prefix + "proj.",
            keep_vars=keep_vars)