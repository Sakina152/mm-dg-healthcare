"""Late-concatenation fusion for ECG signal and patient metadata."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.classifier import ClassifierHead


class FusionClassifier(nn.Module):
    """Fuse two embeddings; either modality may be absent at inference.

    Encoders must return ``(batch, embed_dim)``. Missing modalities are
    represented by learned vectors, allowing full/signal-only/metadata-only
    evaluation with the same trained classifier.
    """
    def __init__(self, signal_encoder: nn.Module, metadata_encoder: nn.Module,
                 fusion_strategy: str = "concat", num_classes: int = 5,
                 signal_embed_dim: int = 128, metadata_embed_dim: int = 128,
                 hidden_dim: int = 64):
        super().__init__()
        if fusion_strategy != "concat":
            raise ValueError("Only fusion_strategy='concat' is implemented and validated.")
        self.signal_encoder = signal_encoder
        self.metadata_encoder = metadata_encoder
        self.fusion_strategy = fusion_strategy
        self.signal_embed_dim = signal_embed_dim
        self.metadata_embed_dim = metadata_embed_dim
        self.missing_signal = nn.Parameter(torch.zeros(1, signal_embed_dim))
        self.missing_metadata = nn.Parameter(torch.zeros(1, metadata_embed_dim))
        nn.init.normal_(self.missing_signal, std=0.02)
        nn.init.normal_(self.missing_metadata, std=0.02)
        self.head = ClassifierHead(signal_embed_dim + metadata_embed_dim, num_classes, hidden_dim)

    def embed(self, signal=None, metadata=None, drop_signal_mask=None, drop_metadata_mask=None):
        reference = signal if signal is not None else metadata
        if reference is None:
            raise ValueError("At least one modality must be supplied to determine batch size/device.")
        batch_size, device = reference.shape[0], reference.device
        if signal is None:
            s = self.missing_signal.expand(batch_size, -1)
        else:
            s = self.signal_encoder(signal)
        if metadata is None:
            m = self.missing_metadata.expand(batch_size, -1)
        else:
            m = self.metadata_encoder(metadata)
        if drop_signal_mask is not None:
            s = torch.where(drop_signal_mask.to(device).bool().view(-1, 1), self.missing_signal.expand(batch_size, -1), s)
        if drop_metadata_mask is not None:
            m = torch.where(drop_metadata_mask.to(device).bool().view(-1, 1), self.missing_metadata.expand(batch_size, -1), m)
        return torch.cat([s, m], dim=1)

    def forward(self, signal=None, metadata=None, drop_signal_mask=None, drop_metadata_mask=None):
        return self.head(self.embed(signal, metadata, drop_signal_mask, drop_metadata_mask))
