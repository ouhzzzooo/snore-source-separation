# src/models/ResUNet1D.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = self.relu(out)
        return out

class ResUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(ResUNet1D, self).__init__()
        features = init_features

        # Encoder
        self.encoder1 = ResidualBlock(in_channels, features, downsample=False)
        self.pool1 = nn.MaxPool1d(2, 2)

        self.encoder2 = ResidualBlock(features, features*2)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.encoder3 = ResidualBlock(features*2, features*4)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.encoder4 = ResidualBlock(features*4, features*8)
        self.pool4 = nn.MaxPool1d(2, 2)

        # Bottleneck
        self.bottleneck = ResidualBlock(features*8, features*16)

        # Decoder
        self.upconv4 = nn.ConvTranspose1d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(features*8*2, features*8)

        self.upconv3 = nn.ConvTranspose1d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(features*4*2, features*4)

        self.upconv2 = nn.ConvTranspose1d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(features*2*2, features*2)

        self.upconv1 = nn.ConvTranspose1d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(features*2, features)

        self.final_conv = nn.Conv1d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc1_pooled = self.pool1(enc1)

        enc2 = self.encoder2(enc1_pooled)
        enc2_pooled = self.pool2(enc2)

        enc3 = self.encoder3(enc2_pooled)
        enc3_pooled = self.pool3(enc3)

        enc4 = self.encoder4(enc3_pooled)
        enc4_pooled = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc4_pooled)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self._align_size(dec4, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._align_size(dec3, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._align_size(dec2, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._align_size(dec1, enc1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.final_conv(dec1)
        return out

    def _align_size(self, x, ref):
        if x.size(-1) != ref.size(-1):
            diff = ref.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[..., :ref.size(-1)]
        return x