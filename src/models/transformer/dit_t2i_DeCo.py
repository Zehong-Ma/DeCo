import torch
import torch.nn as nn

from functools import lru_cache
from src.models.layers.attention_op import attention
from src.models.layers.rope import apply_rotary_emb, precompute_freqs_cis_ex2d as precompute_freqs_cis_2d
from src.models.layers.time_embed import TimestepEmbedder as TimestepEmbedder
from src.models.layers.patch_embed import Embed as Embed
from src.models.layers.swiglu import SwiGLU as FeedForward
from src.models.layers.rmsnorm import RMSNorm as Norm

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_x = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.kv_y = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y, pos) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q.contiguous())
        kx = self.k_norm(kx.contiguous())
        q, kx = apply_rotary_emb(q, kx, freqs_cis=pos)
        kv_y = self.kv_y(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ky, vy = kv_y[0], kv_y[1]
        ky = self.k_norm(ky.contiguous())

        k = torch.cat([kx, ky], dim=2)
        v = torch.cat([vx, vy], dim=2)

        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()

        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4, ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, y, c, pos):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), y, pos)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels+max_freqs**2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        pos = precompute_freqs_cis_2d(self.max_freqs ** 2 * 2, patch_size, patch_size)
        pos = pos[None, :, :].to(device=device, dtype=dtype)
        return pos


    def forward(self, inputs):
        B, P2, C = inputs.shape
        patch_size = int(P2 ** 0.5)
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs

class NerfBlock(nn.Module):
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio=4):
        super().__init__()
        self.param_generator1 = nn.Sequential(
            nn.Linear(hidden_size_s, 2*hidden_size_x**2*mlp_ratio, bias=True),
        )
        self.norm = Norm(hidden_size_x, eps=1e-6)
        self.mlp_ratio = mlp_ratio
    def forward(self, x, s):
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params1 = self.param_generator1(s)
        fc1_param1, fc2_param1 = mlp_params1.chunk(2, dim=-1)
        fc1_param1 = fc1_param1.view(batch_size, hidden_size_x, hidden_size_x*self.mlp_ratio)
        fc2_param1 = fc2_param1.view(batch_size, hidden_size_x*self.mlp_ratio, hidden_size_x)

        # normalize fc1
        normalized_fc1_param1 = torch.nn.functional.normalize(fc1_param1, dim=-2)
        # mlp 1
        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, normalized_fc1_param1)
        x = torch.nn.functional.silu(x)
        x = torch.bmm(x, fc2_param1)
        x = x + res_x
        return x

class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
    def forward(self, x):
        x = self.linear(x)
        return x

class TextRefineAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()
        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4, ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = TextRefineAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.patch_size = patch_size

        self.cond_embed = nn.Linear(z_channels, patch_size**2*model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)
        
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        c = self.cond_embed(c)

        y = c.reshape(c.shape[0], self.patch_size**2, -1)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x)

class PixNerDiT(nn.Module):
    def __init__(
            self,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            decoder_hidden_size=64,
            num_encoder_blocks=18,
            num_decoder_blocks=4,
            num_text_blocks=4,
            patch_size=2,
            txt_embed_dim=1024,
            txt_max_length=100,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.num_blocks = self.num_encoder_blocks + self.num_decoder_blocks
        self.num_text_blocks = num_text_blocks
        self.patch_size = patch_size
        self.txt_embed_dim = txt_embed_dim
        self.txt_max_length = txt_max_length
        self.s_embedder = Embed(in_channels*patch_size**2, hidden_size, bias=True)
        self.x_embedder = NerfEmbedder(in_channels, decoder_hidden_size, max_freqs=8)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = Embed(txt_embed_dim, hidden_size, bias=True, norm_layer=Norm)
        self.y_pos_embedding = torch.nn.Parameter(
            torch.randn(1, txt_max_length, hidden_size),
            requires_grad=True
        )

        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_encoder_blocks)
        ])
        
        self.dec_net = SimpleMLPAdaLN(
            in_channels=self.decoder_hidden_size,
            model_channels=self.decoder_hidden_size,
            out_channels=self.in_channels,  # for vlb loss
            z_channels=self.hidden_size,
            num_res_blocks=self.num_decoder_blocks,
            patch_size=self.patch_size,
            grad_checkpointing=False
        )

        self.text_refine_blocks = nn.ModuleList([
            TextRefineBlock(self.hidden_size, self.num_groups) for _ in range(self.num_text_blocks)
        ])
        self.initialize_weights()
        self.precompute_pos = dict()
        self.weight_path = weight_path
        self.load_ema = load_ema

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, y):
        B, _, H, W = x.shape
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        xpos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        ypos = self.y_pos_embedding
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(B, -1, self.hidden_size) + ypos.to(y.dtype)

        condition = nn.functional.silu(t)
        for i, block in enumerate(self.text_refine_blocks):
            y = block(y, condition)

        s = self.s_embedder(x)
        for i in range(self.num_encoder_blocks):
            s = self.blocks[i](s, y, condition, xpos)

        s = torch.nn.functional.silu(t + s)
        batch_size, length, _ = s.shape
        x = x.reshape(batch_size * length, self.in_channels, self.patch_size ** 2 )
        x = x.transpose(1, 2)
        s = s.view(batch_size * length, self.hidden_size)
        x = self.x_embedder(x)

        x = self.dec_net(x, s)
        
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, length, -1)
        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(),
                                     (H, W),
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size)
        return x