"""Classification head + the Phase-1 baseline model that wraps an encoder.

Kept separate from `SignalEncoder` on purpose: Phase 3's fusion model
reuses this exact `ClassifierHead` on top of a fused embedding instead of
a single-encoder embedding, so the classification logic (and its loss)
never has to be rewritten when fusion is added.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Embedding -> class logits, with an optional hidden layer."""

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int | None = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(embed_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class SignalOnlyClassifier(nn.Module):
    """Phase-1 baseline: SignalEncoder -> ClassifierHead.

    Exposes `.embed(x)` separately from `.forward(x)` so Phase 2's
    rare-class entropy identification can pull embeddings straight out of
    a trained model without needing a separate feature-extraction script.
    """

    def __init__(self, encoder: nn.Module, num_classes: int, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.head = ClassifierHead(embed_dim, num_classes, hidden_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
