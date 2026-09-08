"""Text encoder — SCAFFOLD for Phase 3.

Will wrap a small pretrained transformer (e.g. `distilbert-base-uncased`,
or a biomedical variant like BioClinicalBERT once we compare a couple of
options) via HuggingFace `transformers`, producing a fixed-size embedding
with the same shape convention as `SignalEncoder`:

    TextEncoder.forward(input_ids, attention_mask) -> (batch, embed_dim)

Input text comes from `src.data.ptbxl_text.PTBXLTextLookup.get_text(...)`
(Phase 3, PTB-XL) or later `MIMICCXRTextLookup.get_text(...)` (once
implemented) — the encoder itself doesn't need to know which dataset the
string came from.

Not implemented yet because Phase 3 hasn't started; kept as a stub with
the exact interface so `src/models/fusion.py` can be written and unit
tested against a mock of this class before the real encoder exists.

TODO (Phase 3):
  1. Tokenize with the corresponding HF tokenizer (add a
     `tokenize(texts: list[str]) -> dict` helper here or in a small
     `src/data/text_transforms.py`).
  2. Decide frozen vs. fine-tuned: start frozen (fast, avoids overfitting
     the small PTB-XL text vocabulary) with a linear projection to
     `embed_dim`; compare against fine-tuning as an ablation.
  3. Add a config flag (`model.text_encoder_trainable: bool`) once decided.
"""
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, embed_dim: int):
        super().__init__()
        raise NotImplementedError(
            "TextEncoder is a Phase 3 scaffold. See this module's "
            "docstring for the implementation plan."
        )
