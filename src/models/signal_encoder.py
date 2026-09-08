"""1D-CNN encoder for 12-lead ECG waveforms.

Deliberately simple for Phase 1 (baseline) — a small stack of Conv1d
blocks with batch norm and global average pooling. The point of the
baseline is to establish "how much does plain deep learning suffer under
domain shift," not to chase state-of-the-art ECG architectures; a bigger
model would just make Phase 1 slower to iterate on without changing the
experiment's conclusion.

Outputs a fixed-size embedding vector — the exact interface Phase 3's
fusion module will consume (`embed(x) -> (batch, embed_dim)`), so this
encoder can be reused unchanged inside the fusion model later; only a
second (text) encoder gets added alongside it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SignalEncoder(nn.Module):
    """12-lead ECG -> fixed-size embedding.

    Input:  (batch, 12, signal_length)
    Output: (batch, embed_dim)
    """

    def __init__(
        self,
        in_channels: int = 12,
        embed_dim: int = 128,
        channel_widths: tuple[int, ...] = (32, 64, 128, 128),
    ):
        super().__init__()
        blocks = []
        prev_ch = in_channels
        for width in channel_widths:
            blocks.append(ConvBlock(prev_ch, width))
            prev_ch = width
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(prev_ch, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 12, signal_length) float tensor.

        Returns:
            (batch, embed_dim) embedding.
        """
        h = self.blocks(x)          # (batch, C, L')
        h = self.pool(h).squeeze(-1)  # (batch, C)
        return self.proj(h)          # (batch, embed_dim)
