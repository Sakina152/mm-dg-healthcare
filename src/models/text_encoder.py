"""Legacy text-encoder scaffold; intentionally not used in PTB-XL Phase 3.

PTB-XL text assembled from SCP diagnostic codes leaks information used to
construct the target labels. The current runnable experiment uses
``MetadataEncoder`` over age/sex instead. Do not wire this module into the
PTB-XL label-prediction pipeline unless a genuinely independent text source
and a label-leakage audit are established.
"""
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        raise RuntimeError(
            "TextEncoder is intentionally disabled for PTB-XL because SCP-code-derived "
            "text leaks diagnostic labels. Use MetadataEncoder for the current experiment."
        )
