import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvHead(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(kernel_size=4, in_channels=in_channels, out_channels=hidden_size, stride=2, padding=1),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1), # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1),# 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(kernel_size=1, in_channels=hidden_size, out_channels=1, stride=1, padding=0),  # 1x1 -> 1x1
        )

    def forward(self, feature, text_embedding=None):
        # assume sqrt image size
        B, L, C = feature.shape
        H = W = int(math.sqrt(L))
        feature = feature.permute(0, 2, 1)
        feature = feature.view(B, C, H, W)
        out = self.head(feature).sigmoid().clamp(0.01, 0.99)
        return out

class ConvLinearMMHead(nn.Module):
    def __init__(self, im_channels, mm_channels, hidden_size):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(kernel_size=4, in_channels=im_channels, out_channels=hidden_size, stride=2, padding=1),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1), # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1),# 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear_head = nn.Sequential(
            nn.Linear(mm_channels, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden_size*2, 1)

    def forward(self, im_feature, mm_feature=None):
        # assume sqrt image size
        B, L, C = im_feature.shape
        H = W = int(math.sqrt(L))
        im_feature = im_feature.permute(0, 2, 1)
        im_feature = im_feature.view(B, C, H, W)
        im_out = self.conv_head(im_feature).view(B, -1)
        mm_out = self.linear_head(mm_feature).view(B, -1)
        out = self.out(torch.cat([im_out, mm_out], dim=-1)).sigmoid().clamp(0.01, 0.99)
        return out

class ConvMMHead(nn.Module):
    def __init__(self, im_channels, mm_channels, hidden_size):
        super().__init__()
        self.conv1_head = nn.Sequential(
            nn.Conv2d(kernel_size=4, in_channels=im_channels, out_channels=hidden_size, stride=2, padding=1),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1), # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1),# 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2_head = nn.Sequential(
            nn.Conv2d(kernel_size=4, in_channels=mm_channels, out_channels=hidden_size, stride=2, padding=1),
            # 16x16 -> 8x8
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1),
            # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1),
            # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=hidden_size),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out = nn.Linear(hidden_size*2, 1)

    def forward(self, im_feature, mm_feature=None):
        # assume sqrt image size
        B, L, C = im_feature.shape
        H = W = int(math.sqrt(L))
        im_feature = im_feature.permute(0, 2, 1)
        im_feature = im_feature.view(B, C, H, W)

        B, Lmm, Cmm = mm_feature.shape
        Hmm = Wmm = int(math.sqrt(Lmm))
        mm_feature = mm_feature.permute(0, 2, 1)
        mm_feature = mm_feature.view(B, Cmm, Hmm, Wmm)

        im_out = self.conv1_head(im_feature).view(B, -1)
        mm_out = self.conv2_head(mm_feature).view(B, -1)
        out = self.out(torch.cat([im_out, mm_out], dim=-1)).sigmoid().clamp(0.01, 0.99)
        return out

# class ConvTextHead(nn.Module):
#     def __init__(self, in_channels, text_channels, hidden_size):
#         super().__init__()
#         self.head = nn.Sequential(
#             nn.Conv2d(kernel_size=4, in_channels=in_channels, out_channels=hidden_size, stride=2, padding=1),  # 16x16 -> 8x8
#             nn.GroupNorm(num_groups=32, num_channels=hidden_size),
#             nn.SiLU(),
#             nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1), # 8x8 -> 4x4
#             nn.GroupNorm(num_groups=32, num_channels=hidden_size),
#             nn.SiLU(),
#             nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1),# 8x8 -> 4x4
#             nn.GroupNorm(num_groups=32, num_channels=hidden_size),
#             nn.SiLU(),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(kernel_size=1, in_channels=hidden_size, out_channels=hidden_size, stride=1, padding=0),  # 1x1 -> 1x1
#         )
#         self.text_head = nn.Sequential(
#             nn.Linear(text_channels, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#         )
#
#     def forward(self, feature, text_embedding=None):
#         # assume sqrt image size
#         B, L, C = feature.shape
#         H = W = int(math.sqrt(L))
#         feature = feature.permute(0, 2, 1)
#         feature = feature.view(B, C, H, W)
#         feature = self.head(feature).view(B, -1)
#         text_embedding = torch.mean(text_embedding, dim=1, keepdim=False)
#         text_embedding = self.text_head(text_embedding)
#         logits = torch.sum(feature * text_embedding, dim=1, keepdim=False)
#         score = logits.sigmoid().clamp(0.01, 0.99)
#         return score
#
# class LinearHead(nn.Module):
#     def __init__(self, in_channels, hidden_size):
#         super().__init__()
#         self.head = nn.Sequential(
#             nn.Linear(in_channels, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, 1),
#         )
#     def forward(self, feature, text_embedding=None):
#         out = self.head(feature).sigmoid().clamp(0.01, 0.99)
#         return out


# class ConvMultiModalHead(nn.Module):
#     def __init__(self, in_channels, mm_channels, hidden_size):
#         super().__init__()
#         self.image_head = nn.Sequential(
#             nn.Conv2d(kernel_size=4, in_channels=in_channels, out_channels=hidden_size, stride=2, padding=1),  # 16x16 -> 8x8
#             nn.GroupNorm(num_groups=32, num_channels=hidden_size),
#             nn.SiLU(),
#             nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1), # 8x8 -> 4x4
#             nn.GroupNorm(num_groups=32, num_channels=hidden_size),
#             nn.SiLU(),
#             nn.Conv2d(kernel_size=4, in_channels=hidden_size, out_channels=hidden_size, stride=2, padding=1),# 8x8 -> 4x4
#             nn.GroupNorm(num_groups=32, num_channels=hidden_size),
#             nn.SiLU(),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(kernel_size=1, in_channels=hidden_size, out_channels=1, stride=1, padding=0),  # 1x1 -> 1x1
#         )
#         self.mm_head = nn.Sequential(
#             nn.Linear(mm_channels, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#         )
#
#     def forward(self, feature, text_embedding=None):
#         # assume sqrt image size
#         B, L, C = feature.shape
#         H = W = int(math.sqrt(L))
#         feature = feature.permute(0, 2, 1)
#         feature = feature.view(B, C, H, W)
#         feature = self.head(feature).view(B, -1)
#         text_embedding = torch.mean(text_embedding, dim=1, keepdim=False)
#         text_embedding = self.text_head(text_embedding)
#         logits = torch.sum(feature * text_embedding, dim=1, keepdim=False)
#         score = logits.sigmoid().clamp(0.01, 0.99)
#         return score

# class TransformerTextHead(nn.Module):
#     def __init__(self, in_channels, text_channels, hidden_size):
#         super().__init__()
#
#         self.transformer = nn.Sequential(
#             nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size, batch_first=True),
#             nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size, batch_first=True),
#             nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size, batch_first=True),
#             nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size, batch_first=True),
#         )
#         self.text_head = nn.Sequential(
#             nn.Linear(text_channels, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#         )
#         self.feature_head = nn.Sequential(
#             nn.Linear(in_channels, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#         )
#         self.cls_head = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, 1),
#         )
#
#     def forward(self, feature, text_embedding=None):
#         # assume sqrt image size
#         feature = self.feature_head(feature)
#         text_embedding = self.text_head(text_embedding)
#         tokens = torch.cat([feature, text_embedding], dim=1)
#         tokens = self.transformer(tokens)
#         cls_token = tokens
#         logits = self.cls_head(cls_token)
#         logits = torch.mean(logits, dim=1, keepdim=False)
#         score = logits.sigmoid().clamp(0.01, 0.99)
#         return score
