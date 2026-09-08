"""Modality dropout — SCAFFOLD for Phase 5.

Implements the proposal's Section 6.2 step 5: during training, randomly
zero out (or replace with the learned "missing" embedding — see
`fusion.py`'s TODO) one modality per sample, so `FusionClassifier` learns
to rely on whichever modality is actually present instead of assuming
both always are. This is what lets us later test "signal missing" and
"text missing" conditions on the target domain and report the degradation
in `src.evaluation.metrics.modality_degradation`.

Planned interface:

    class ModalityDropout:
        def __init__(self, p_drop_signal: float, p_drop_text: float): ...
        def __call__(self, batch: dict) -> dict:
            # returns batch with "signal" and/or "input_ids"/"attention_mask"
            # replaced by a sentinel (None, or a masking flag) per-sample,
            # according to p_drop_signal / p_drop_text, mutually exclusive
            # per sample (never drop both — see fusion_robust.yaml's note).

Depends on `FusionClassifier` (fusion.py) supporting `signal=None` /
`input_ids=None` inputs, which is itself still a Phase 3 TODO — so this
module cannot be implemented before Phase 3 is done.

TODO (Phase 5):
  1. Implement `ModalityDropout.__call__` as a batch transform applied
     inside the Phase-3 training loop (extend `src/training/trainer.py`'s
     `run_epoch`, gated by `cfg.modality_dropout.enabled`).
  2. Implement the three-condition eval loop described in
     `configs/fusion_robust.yaml`'s `eval.conditions`
     (full / signal_only / text_only), reusing
     `src.evaluation.metrics.modality_degradation` to report the drop.
  3. Unit test the dropout rates empirically (e.g. over 10,000 synthetic
     samples, check observed drop proportions are within a tolerance of
     `p_drop_signal` / `p_drop_text`) — no real data needed for this test,
     same pattern as `tests/test_entropy.py`.
"""


class ModalityDropout:
    def __init__(self, p_drop_signal: float, p_drop_text: float):
        raise NotImplementedError(
            "ModalityDropout is a Phase 5 scaffold. See this module's "
            "docstring for the implementation plan. Depends on "
            "FusionClassifier (src/models/fusion.py) supporting missing-"
            "modality inputs first."
        )
