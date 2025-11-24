import torch
import torch.nn as nn

class _SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim*2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(torch.nn.functional.silu(x1)*x2)


# try:
# from xformers.ops import SwiGLU as aa
#     SwiGLU = SwiGLU
#     print("use xformers swiglu")
# except:
#     print("use slow swiglu")

SwiGLU = _SwiGLU