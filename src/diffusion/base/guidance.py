import torch

def simple_guidance_fn(out, cfg):
    uncondition, condtion = out.chunk(2, dim=0)
    out = uncondition + cfg * (condtion - uncondition)
    return out

def guidance_fn_with_rescale(out, cfg, rescale_factor=0.7):
    """
    对模型的原始输出应用Classifier-Free Guidance (CFG)，并加入方差重缩放 (rescale_cfg)。

    Args:
        out (torch.Tensor): 模型的原始输出，包含了unconditional和conditional两部分。
        cfg (float): Guidance scale，即引导强度。
        rescale_factor (float): 重缩放因子。常用的值在0.5到0.8之间，0.7是一个很好的起始值。

    Returns:
        torch.Tensor: 应用了CFG和rescale_cfg之后的最终输出。
    """
    uncondition, condition = out.chunk(2, dim=0)

    guided_out = uncondition + cfg * (condition - uncondition)

    std_condition = torch.std(condition, dim=(1,2,3), keepdim=True)
    std_guided = torch.std(guided_out, dim=(1,2,3), keepdim=True)
    
    scale = std_condition / (std_guided + 1e-6)
    print(scale.mean())
    rescaled_out = guided_out * (scale * rescale_factor + 1.0 * (1.0 - rescale_factor))
    return rescaled_out

def c3_guidance_fn(out, cfg):
    # guidance function in DiT/SiT, seems like a bug not a feature?
    uncondition, condtion = out.chunk(2, dim=0)
    out = condtion
    out[:, :3] = uncondition[:, :3] + cfg * (condtion[:, :3] - uncondition[:, :3])
    return out