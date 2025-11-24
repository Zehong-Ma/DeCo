import torch

@torch.no_grad()
def copy_params(src_model, dst_model):
    for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
        dst_param.data.copy_(src_param.data)

@torch.no_grad()
def swap_tensors(tensor1, tensor2):
    tmp = torch.empty_like(tensor1)
    tmp.copy_(tensor1)
    tensor1.copy_(tensor2)
    tensor2.copy_(tmp)