import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ensure_equal_shape(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    padding = [0, 0, 0, 0]
    if x.shape[-1] != y.shape[-1]:
        padding[1] = 1  # Padding right
    if x.shape[-2] != y.shape[-2]:
        padding[3] = 1  # Padding bottom
    if torch.tensor(padding).sum().item() != 0:  # PyTorch 연산으로 변경
        x = F.pad(x, padding, "reflect")
    return x


def sinusoidal_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution to reduce computation cost."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientUNetBlock(nn.Module):
    """A single block in the U-Net architecture with DepthwiseSeparableConv and residual connection."""

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(EfficientUNetBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.norm = nn.InstanceNorm2d(out_channels)  # GroupNorm with 8 groups
        self.relu = nn.GELU()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        self.time_mlp1 = nn.Linear(time_embedding_dim, in_channels)
        self.time_mlp2 = nn.Linear(in_channels, out_channels)

    def forward(self, x, t_embed):
        identity = x

        t1 = self.time_mlp1(t_embed)
        t2 = self.time_mlp2(t1)

        x = self.relu((self.conv1(x + t1[:, :, None, None])))
        x = self.relu((self.conv2(x + t2[:, :, None, None])))

        if self.residual is not None:
            identity = self.residual(identity)

        return x + identity  # Residual connection


class EfficientUNet(nn.Module):
    """Efficient U-Net with Depthwise Separable Convolution and GroupNorm."""

    def __init__(self, in_channels=2, out_channels=1, init_features=32):
        super(EfficientUNet, self).__init__()

        features = init_features
        self.time_embedding_dim = init_features
        # Down-sampling path (Encoder)
        self.encoder1 = EfficientUNetBlock(in_channels, features, self.time_embedding_dim)
        self.encoder2 = EfficientUNetBlock(features, features * 2, self.time_embedding_dim)
        self.encoder3 = EfficientUNetBlock(features * 2, features * 4, self.time_embedding_dim)
        self.encoder4 = EfficientUNetBlock(features * 4, features * 8, self.time_embedding_dim)

        # Average Pooling layers for down-sampling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Bottom block
        self.bottom = EfficientUNetBlock(features * 8, features * 32, self.time_embedding_dim)

        # Up-sampling path (Decoder)
        self.upsample = nn.PixelShuffle(2)
        self.decoder4 = EfficientUNetBlock(features * 8 + features * 8, features * 16, self.time_embedding_dim)  # Concat encoder4 output
        self.decoder3 = EfficientUNetBlock(features * 4 + features * 4, features * 8, self.time_embedding_dim)  # Concat encoder3 output
        self.decoder2 = EfficientUNetBlock(features * 2 + features * 2, features * 4, self.time_embedding_dim)  # Concat encoder2 output
        self.decoder1 = EfficientUNetBlock(features * 1 + features * 1, features * 2, self.time_embedding_dim)  # Concat encoder1 output

        # Output layer
        self.output_conv = nn.Conv2d(features * 2, out_channels, kernel_size=1)

    def forward(self, x_n, x_c, t):
        t_embed = sinusoidal_embedding(t, self.time_embedding_dim).to(x_n.device)
        x = torch.cat([x_n, x_c], 1)
        # Encoder
        enc1 = self.encoder1(x, t_embed)
        enc2 = self.encoder2(self.pool(enc1), t_embed)
        enc3 = self.encoder3(self.pool(enc2), t_embed)
        enc4 = self.encoder4(self.pool(enc3), t_embed)

        # Bottom
        bottom = self.bottom(self.pool(enc4), t_embed)

        # Decoder with skip connections
        dec4 = self.upsample(bottom)
        dec4 = ensure_equal_shape(dec4, enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.decoder4(dec4, t_embed)

        dec3 = self.upsample(dec4)
        dec3 = ensure_equal_shape(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3, t_embed)

        dec2 = self.upsample(dec3)
        dec2 = ensure_equal_shape(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2, t_embed)

        dec1 = self.upsample(dec2)
        dec1 = ensure_equal_shape(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1, t_embed)

        return self.output_conv(dec1)
