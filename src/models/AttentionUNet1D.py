# src/models/AttentionUNet1D.py

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch, embed_dim, seq_len]
        # Permute to [seq_len, batch, embed_dim] for multihead attention
        x = x.permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        x = self.dropout(x)
        # Permute back to [batch, embed_dim, seq_len]
        return x.permute(1, 2, 0)

class SpectralAttentionBlock(nn.Module):
    def __init__(self, embed_dim):
        super(SpectralAttentionBlock, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, embed_dim, seq_len]
        spectral_attention = self.sigmoid(self.conv1(x))
        x = x * spectral_attention
        x = self.conv2(x)
        return x

class AttentionUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, num_heads=8):
        super(AttentionUNet1D, self).__init__()

        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features*2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features*2, features*4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features*4, features*8)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features*8, features*16)

        # Decoder
        self.upconv4 = nn.ConvTranspose1d(features*16, features*8, kernel_size=2, stride=2)
        self.att4 = MultiHeadAttentionBlock(embed_dim=features*8, num_heads=num_heads)
        self.spec_att4 = SpectralAttentionBlock(embed_dim=features*8)
        self.decoder4 = self._block(features*8 * 2, features*8)

        self.upconv3 = nn.ConvTranspose1d(features*8, features*4, kernel_size=2, stride=2)
        self.att3 = MultiHeadAttentionBlock(embed_dim=features*4, num_heads=num_heads)
        self.spec_att3 = SpectralAttentionBlock(embed_dim=features*4)
        self.decoder3 = self._block(features*4 * 2, features*4)

        self.upconv2 = nn.ConvTranspose1d(features*4, features*2, kernel_size=2, stride=2)
        self.att2 = MultiHeadAttentionBlock(embed_dim=features*2, num_heads=num_heads)
        self.spec_att2 = SpectralAttentionBlock(embed_dim=features*2)
        self.decoder2 = self._block(features*2 * 2, features*2)

        self.upconv1 = nn.ConvTranspose1d(features*2, features, kernel_size=2, stride=2)
        self.att1 = MultiHeadAttentionBlock(embed_dim=features, num_heads=num_heads)
        self.spec_att1 = SpectralAttentionBlock(embed_dim=features)
        self.decoder1 = self._block(features*2, features)

        self.conv = nn.Conv1d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = checkpoint.checkpoint(self.encoder1, x)
        enc2 = checkpoint.checkpoint(self.encoder2, self.pool1(enc1))
        enc3 = checkpoint.checkpoint(self.encoder3, self.pool2(enc2))
        enc4 = checkpoint.checkpoint(self.encoder4, self.pool3(enc3))

        bottleneck = checkpoint.checkpoint(self.bottleneck, self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.att4(dec4)
        dec4 = self.spec_att4(dec4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = checkpoint.checkpoint(self.decoder4, dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.att3(dec3)
        dec3 = self.spec_att3(dec3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = checkpoint.checkpoint(self.decoder3, dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.att2(dec2)
        dec2 = self.spec_att2(dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = checkpoint.checkpoint(self.decoder2, dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.att1(dec1)
        dec1 = self.spec_att1(dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = checkpoint.checkpoint(self.decoder1, dec1)

        out = self.conv(dec1)
        return out

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )