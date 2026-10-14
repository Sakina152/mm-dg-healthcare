"""Small encoder for non-diagnostic PTB-XL patient metadata.

The primary Phase-3 experiment uses age and sex only. Diagnostic codes and
report-derived fields are intentionally excluded because they reveal the target.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, embed_dim: int = 128, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        if metadata.ndim != 2:
            raise ValueError(f"metadata must have shape (batch, features), got {tuple(metadata.shape)}")
        return self.net(metadata.float())
