import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):
    """
    Applies multi-head self-attention along the (batch, embed_dim, length) dimension,
    treating 'embed_dim' as the 'channel' dimension. We permute it to
    [sequence_length, batch, embed_dim] for PyTorch's MultiheadAttention,
    then return it back to [batch, embed_dim, length].
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x shape: [batch, embed_dim, length]
        # Permute -> [length, batch, embed_dim]
        x = x.permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)  # self-attention
        x = self.norm(x + attn_output)
        x = self.dropout(x)
        # Permute back to [batch, embed_dim, length]
        return x.permute(1, 2, 0)


class SpectralAttentionBlock(nn.Module):
    """
    A simple spectral attention block:
      1) Generate an attention mask via sigmoid(Conv1d)
      2) Multiply input by that mask
      3) Another Conv1d for final transform
    """
    def __init__(self, embed_dim):
        super(SpectralAttentionBlock, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, embed_dim, length]
        # produce attention map
        attn_map = self.sigmoid(self.conv1(x))
        # apply
        x = x * attn_map
        # final conv
        x = self.conv2(x)
        return x


class AttentionUNet1D(nn.Module):
    """
    A 1D U-Net with multi-head attention + spectral attention at each decoder stage.
    init_features controls the base channel count.
    Each encoder doubles the features, each decoder halves.
    num_heads must divide (features * 2^level).
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32, num_heads=4):
        super(AttentionUNet1D, self).__init__()

        # initial channel count
        features = init_features

        # ---------------- Encoder ----------------
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ---------------- Bottleneck ----------------
        self.bottleneck = self._block(features * 8, features * 16)

        # ---------------- Decoder ----------------
        # 1) Upconv + attention + cat skip + conv block
        self.upconv4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2)
        self.att4 = MultiHeadAttentionBlock(embed_dim=features * 8, num_heads=num_heads)
        self.spec_att4 = SpectralAttentionBlock(embed_dim=features * 8)
        self.decoder4 = self._block((features * 8) * 2, features * 8)

        self.upconv3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = MultiHeadAttentionBlock(embed_dim=features * 4, num_heads=num_heads)
        self.spec_att3 = SpectralAttentionBlock(embed_dim=features * 4)
        self.decoder3 = self._block((features * 4) * 2, features * 4)

        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = MultiHeadAttentionBlock(embed_dim=features * 2, num_heads=num_heads)
        self.spec_att2 = SpectralAttentionBlock(embed_dim=features * 2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)

        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.att1 = MultiHeadAttentionBlock(embed_dim=features, num_heads=num_heads)
        self.spec_att1 = SpectralAttentionBlock(embed_dim=features)
        self.decoder1 = self._block(features * 2, features)

        self.final_conv = nn.Conv1d(features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x shape: [batch, in_channels, length]
        returns: [batch, out_channels, length_out]
        """
        # -------- Encoders --------
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # -------- Bottleneck --------
        bottleneck = self.bottleneck(self.pool4(enc4))

        # -------- Decoder 4 --------
        dec4 = self.upconv4(bottleneck)
        dec4 = self.att4(dec4)
        dec4 = self.spec_att4(dec4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        # -------- Decoder 3 --------
        dec3 = self.upconv3(dec4)
        dec3 = self.att3(dec3)
        dec3 = self.spec_att3(dec3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        # -------- Decoder 2 --------
        dec2 = self.upconv2(dec3)
        dec2 = self.att2(dec2)
        dec2 = self.spec_att2(dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        # -------- Decoder 1 --------
        dec1 = self.upconv1(dec2)
        dec1 = self.att1(dec1)
        dec1 = self.spec_att1(dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # final conv
        out = self.final_conv(dec1)
        return out

    @staticmethod
    def _block(in_channels, out_channels):
        """
        Standard two-conv block used in each stage.
        """
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )