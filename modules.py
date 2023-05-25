import torch
from torch import nn
import torchvision.models as models
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.cnn(x))


class GhostModuleM(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, stride=1, ratio=2, dw_size=3, dropout=0.2, use_batchnorm=True):
        super(GhostModuleM, self).__init__()
        self.oup = oup
        self.use_batchnorm = use_batchnorm

        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv1 = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation1 = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.primary_conv2 = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.InstanceNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation2 = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.InstanceNorm2d(new_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.use_batchnorm:
            x1 = self.primary_conv1(x)
            x2 = self.cheap_operation1(x1)
            out = torch.cat([x1, x2], dim=1)
        else:
            x1 = self.primary_conv2(x)
            x2 = self.cheap_operation2(x1)
            out = torch.cat([x1, x2], dim=1)

        return out[:, :self.oup, :, :]


class UpConvM(nn.Module):
    def __init__(self, in_channels, out_channels, mode='nearest', use_batchnorm=True):
        super(UpConvM, self).__init__()

        self.use_batchnorm = use_batchnorm

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.use_batchnorm:
            x = self.up1(x)
        else:
            x = self.up2(x)

        return x
