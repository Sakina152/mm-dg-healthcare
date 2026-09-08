"""Multimodal fusion — SCAFFOLD for Phase 3.

This is the module that actually implements the proposal's core novelty:
combining `SignalEncoder`'s embedding with `TextEncoder`'s embedding
before the shared `ClassifierHead` (see `src/models/classifier.py`,
already implemented and reused here unchanged).

Planned interface (mirrors `SignalOnlyClassifier` in classifier.py so
`src/training/trainer.py`'s loop needs only a `build_model`/`build_data`
swap, not a rewrite, to go from Phase 1 to Phase 3):

    class FusionClassifier(nn.Module):
        def __init__(self, signal_encoder, text_encoder, fusion_strategy, num_classes, ...): ...
        def embed(self, signal, input_ids, attention_mask) -> Tensor: ...
        def forward(self, signal, input_ids, attention_mask) -> Tensor: ...

Two fusion strategies to implement and compare (config: `model.fusion_strategy`):

  - "concat": concatenate signal_embed and text_embed, project down with
    a linear layer. Simple, strong baseline for fusion — implement first.
  - "cross_attention": text embedding attends over signal embedding (or
    vice versa) before pooling. More expressive, higher risk of
    overfitting on PTB-XL's relatively small text vocabulary (coded
    statements, not free text) — worth trying once "concat" works, not
    before.

Missing-modality support (needed by Phase 5, `modality_dropout.py`): both
`embed` and `forward` should accept `signal=None` or `input_ids=None` and
substitute a learned "missing modality" embedding (a single trainable
vector, not a zero vector — zeros are indistinguishable from a genuine
all-zero signal and would confuse the fusion layer) instead of crashing.

TODO (Phase 3):
  1. Implement `FusionClassifier` with `fusion_strategy="concat"` first.
  2. Unit test with mock encoders (small random-weight `nn.Linear`
     stand-ins) so the fusion arithmetic can be verified without a real
     dataset or trained text encoder — mirrors how `tests/test_entropy.py`
     tests the math independent of any real embeddings.
  3. Wire into `src/training/trainer.py`'s `build_model`/`build_data`
     once `TextEncoder` (see text_encoder.py) is implemented.
  4. Add "cross_attention" as a second strategy once "concat" is verified
     end-to-end.
"""
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(self, signal_encoder, text_encoder, fusion_strategy: str, num_classes: int, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "FusionClassifier is a Phase 3 scaffold. See this module's "
            "docstring for the implementation plan. Depends on "
            "TextEncoder (src/models/text_encoder.py) being implemented "
            "first."
        )
