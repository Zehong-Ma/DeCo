import torch

@torch.no_grad()
def no_grad(net):
    assert net is not None, "net is None"
    for param in net.parameters():
        param.requires_grad = False
    net.eval()
    return net

@torch.no_grad()
def filter_nograd_tensors(params_list):
    filtered_params_list = []
    for param in params_list:
        if param.requires_grad:
            filtered_params_list.append(param)
    return filtered_params_list