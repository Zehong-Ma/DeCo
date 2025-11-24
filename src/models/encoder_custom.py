import copy
import torch
import torch.nn as nn
import timm
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import os

class IndentityMapping(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        return x

class DINOv2(nn.Module):
    def __init__(self, weight_path:str, base_patch_size=16):
        super(DINOv2, self).__init__()
        # directory = os.path.dirname(weight_path)
        # weight_path = os.path.basename(weight_path)
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # self.encoder = torch.hub.load(
        #     directory,
        #     weight_path,
        #     source="local",
        #     skip_validation=True
        # )
        self.encoder = self.encoder.to(torch.bfloat16)
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()
        self.base_patch_size = base_patch_size
        self.encoder.compile()

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if resize:
            x = torch.nn.functional.interpolate(x, (int(14*h/self.base_patch_size), int(14*w/self.base_patch_size)), mode='bicubic')
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        feature = feature.to(torch.bfloat16)
        return feature

from transformers import CLIPModel, CLIPTokenizer
class CLIP(nn.Module):
    def __init__(self, weight_path:str):
        super(CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained(weight_path).to(torch.bfloat16)
        self.tokenizer = CLIPTokenizer.from_pretrained(weight_path)
        self.height = self.model.config.vision_config.image_size
        self.width = self.model.config.vision_config.image_size

        self.model.vision_model.compile()
        self.model.text_model.compile()
    def forward(self, x, text, resize=True):
        tokens = self.tokenizer(text, truncation=True, return_tensors='pt', padding="max_length", max_length=self.tokenizer.model_max_length).input_ids.cuda()
        text_output = self.model.text_model(input_ids=tokens).last_hidden_state
        text_output = self.model.text_projection(text_output)
        text_output = torch.nn.functional.normalize(text_output, dim=-1, p=2)
        if resize:
            x = torch.nn.functional.interpolate(x, (self.height, self.width), mode='bicubic')
        x = Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)(x)
        vision_output = self.model.vision_model(x).last_hidden_state[:, 1:]
        vision_output = self.model.visual_projection(vision_output)
        vision_output = torch.nn.functional.normalize(vision_output, dim=-1, p=2)
        output = torch.bmm(vision_output, text_output.transpose(1, 2))
        return output

from transformers import SiglipModel, GemmaTokenizer, SiglipTokenizer
class SigLIP(nn.Module):
    def __init__(self, weight_path:str):
        super(SigLIP, self).__init__()
        if "siglip2" in weight_path:
            self.tokenizer = GemmaTokenizer.from_pretrained(weight_path)
        else:
            self.tokenizer = SiglipTokenizer.from_pretrained(weight_path)
        self.model = SiglipModel.from_pretrained(weight_path).to(torch.bfloat16)

        self.mean = 0.5
        self.std = 0.5

        self.model.vision_model.compile()
        self.model.text_model.compile()
    def forward(self, x, text, resize=True):
        tokens = self.tokenizer(text, truncation=True, return_tensors='pt', padding="max_length", max_length=64).input_ids.cuda()
        text_output = self.model.text_model(input_ids=tokens).last_hidden_state
        text_output = torch.nn.functional.normalize(text_output, dim=-1, p=2)
        if resize:
            x = torch.nn.functional.interpolate(x, (self.height, self.width), mode='bicubic')
        x = (x - self.mean)/self.std
        vision_output = self.model.vision_model(x).last_hidden_state
        vision_output = torch.nn.functional.normalize(vision_output, dim=-1, p=2)
        output = torch.bmm(vision_output, text_output.transpose(1, 2))
        return output

from transformers import SiglipVisionModel
class SigLIPVision(nn.Module):
    def __init__(self, weight_path:str, base_patch_size=16):
        super(SigLIPVision, self).__init__()
        self.model = SiglipVisionModel.from_pretrained(weight_path).to(torch.bfloat16)
        self.height = self.model.config.image_size
        self.width = self.model.config.image_size
        self.patch_size = self.model.config.patch_size
        self.base_patch_size = base_patch_size
        self.model.compile()
        self.mean = 0.5
        self.std = 0.5
    def forward(self, x, resize=True):
        if resize:
            h, w = x.shape[-2:]
            new_h = int(self.patch_size * h / self.base_patch_size)
            new_w = int(self.patch_size * w / self.base_patch_size)
            x = torch.nn.functional.interpolate(x, (new_h, new_w), mode='bicubic')
        x = (x - self.mean)/self.std
        vision_output = self.model.vision_model(x).last_hidden_state
        return vision_output