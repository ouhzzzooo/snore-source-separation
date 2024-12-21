# src/models/WaveUNet1D.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_levels=6, init_features=64):
        super(WaveUNet1D, self).__init__()

        self.num_levels = num_levels
        features = init_features

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        encoder_features = []

        current_in_channels = in_channels
        for _ in range(num_levels):
            self.encoders.append(self._down_block(current_in_channels, features))
            encoder_features.append(features)
            self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            current_in_channels = features
            features *= 2  # Double number of features

        # Bottleneck
        self.bottleneck = self._down_block(current_in_channels, features)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        current_in_channels = features
        for i in range(num_levels):
            features //= 2
            self.upconvs.append(
                nn.ConvTranspose1d(current_in_channels, features, kernel_size=2, stride=2)
            )
            decoder_in_channels = features + encoder_features[-(i + 1)]
            self.decoders.append(self._up_block(decoder_in_channels, features))
            current_in_channels = features

        self.final_conv = nn.Conv1d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc_outputs = []

        # Encoder
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            enc_outputs.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.num_levels):
            x = self.upconvs[i](x)
            # handle size mismatch
            enc_output = enc_outputs[-(i + 1)]
            x = self._pad_or_trim(x, enc_output.size(-1))
            x = torch.cat((x, enc_output), dim=1)
            x = self.decoders[i](x)

        return self.final_conv(x)

    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _pad_or_trim(self, x, target_size):
        diff = target_size - x.size(-1)
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[..., :target_size]
        return x