# -*- coding: utf-8 -*-
# @Author: Chen Renjie
# @Date:   2021-04-10 19:49:35
# @Last Modified by:   Chen Renjie
# @Last Modified time: 2021-05-30 00:44:05

"""ResNet for CIFAR10
Change conv1 kernel size from 7 to 3
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

class MSA(nn.Module):
    def __init__(self, window_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B_, N, C = x.shape
        num_heads = self.num_heads
        window_size = self.window_size

        qkv = self.qkv(x).reshape(B_, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(window_size * window_size, window_size * window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ConvformerLayer(nn.Module):
    def __init__(self, 
        window_size, resolution, embed_dim, num_heads, 
        mlp_ratio=2., qkv_bias=False, qk_scale=None, 
        drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
    ):
        super(ConvformerLayer, self).__init__()

        k, p, s = window_size, 0, window_size
        l = resolution
        self.k, self.p, self.s = k, p, s
        self.num_wins = ((l - k + 2 * p ) // s + 1)**2

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MSA(window_size, embed_dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop_ratio)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.unfold = nn.Unfold(kernel_size=k, padding=0, stride=s)
        self.fold = nn.Fold(output_size=l, kernel_size=k, padding=0, stride=s)
        
        scaler = torch.ones(1, embed_dim, resolution, resolution)
        scaler = self.fold(self.unfold(scaler))

        self.register_buffer("scaler", scaler)

    def forward(self, x):
        B, C, H, W = x.shape
        num_wins = self.num_wins
        k = self.k
        x = self.unfold(x)
        x = x.permute(0, 2, 1).reshape(B*num_wins, C, k*k).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1).reshape(B, num_wins, C*k*k).permute(0, 2, 1)
        x = self.fold(x)
        x = x / self.scaler
        return x
        
class Convformer(nn.Module):
    def __init__(self, num_classes, img_size=32, embed_dim=96, window_size=4, num_blocks=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super(Convformer, self).__init__()
        self.img_size = img_size
        self.window_size = window_size
        self.channels = embed_dim
        self.stage0 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=3, stride=1, padding=1, bias=False)),
            # ("bn", nn.BatchNorm2d(num_features=self._C[1])),
            # ("relu", nn.ReLU(inplace=True))
        ]))

        self.stage1 = self.make_layer(1*embed_dim, num_blocks[0], window_size, img_size//1, num_heads[0], 0.025, 0)
        self.stage2 = self.make_layer(2*embed_dim, num_blocks[1], window_size, img_size//2, num_heads[1], 0.050, 2)
        self.stage3 = self.make_layer(4*embed_dim, num_blocks[2], window_size, img_size//4, num_heads[2], 0.075, 2)
        self.stage4 = self.make_layer(8*embed_dim, num_blocks[3], window_size, img_size//8, num_heads[3], 0.100, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self.channels, num_classes)

    def make_layer(self, out_channels, block_num, window_size, resolution, num_heads, drop_path_ratio, downsample=0):
        layers = OrderedDict()
        layers["downsample"] = nn.Conv2d(self.channels, out_channels, downsample, downsample, 0) if downsample>0 else nn.Identity()
        self.channels = out_channels
        for i in range(block_num):
            layers[f"block{i+1}"] = ConvformerLayer(window_size, resolution, self.channels, num_heads, drop_path_ratio=drop_path_ratio)
        return nn.Sequential(layers)



    def forward(self, x):
        x = self.stage0(x)     # (N,  64, 32, 32)
        x = self.stage1(x)     # (N,  64, 32, 32)
        x = self.stage2(x)     # (N, 128, 16, 16)
        x = self.stage3(x)     # (N, 256,  8,  8)
        x = self.stage4(x)     # (N, 512,  4,  4)
        x = self.avgpool(x)    # (N, 512,  1,  1)
        x = self.flatten(x)    # (N, 512)
        x = self.fc(x)         # (N, 10)
        return x

def convformerTiny(num_classes):
    return Convformer(num_classes, img_size=32, embed_dim=64, window_size=4, num_blocks=[1, 1, 2, 1], num_heads=[2, 4, 8, 16])

def convformerSmall(num_classes):
    return Convformer(num_classes, img_size=32, embed_dim=64, window_size=4, num_blocks=[1, 2, 2, 2], num_heads=[2, 4, 8, 16])
    # return Convformer(num_classes, img_size=32, embed_dim=96, window_size=4, num_blocks=[1, 1, 2, 1], num_heads=[3, 6, 12, 24])

def convformerBase(num_classes):
    return Convformer(num_classes, img_size=32, embed_dim=128, window_size=4, num_blocks=[2, 2, 4, 2], num_heads=[4, 8, 16, 32])

# def convformerBase(num_classes):
#     return Convformer(num_classes, img_size=32, embed_dim=192, window_size=4, num_blocks=[4, 4, 12, 4], num_heads=[6, 12, 24, 48])

# unit test
if __name__ == "__main__":
    import time
    num_classes = 10
    x = torch.rand(1, 3, 32, 32)
    for arch in [convformerTiny, convformerSmall, convformerBase]:
        model = arch(num_classes).eval()
        params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            ts = time.time()
            out = model(x)
            t = time.time() - ts
        
        print(f"{arch.__name__} params: {params/1e6:.2f}M Infer: {t:.3f}s")