"""Metrics matching the proposal's Section 6.3.

Overall accuracy/macro-F1 are reported for comparability with prior
published baselines, but the PRIMARY metric throughout this project is
rare-class F1/precision/recall — overall accuracy can look fine while a
model completely misses the class that matters most clinically, which is
exactly the failure mode the reference paper (and this project) cares
about.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    rare_class_labels: list
    rare_class_f1: float
    rare_class_precision: float
    rare_class_recall: float
    per_class_f1: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "rare_class_labels": self.rare_class_labels,
            "rare_class_f1": self.rare_class_f1,
            "rare_class_precision": self.rare_class_precision,
            "rare_class_recall": self.rare_class_recall,
            "per_class_f1": self.per_class_f1,
        }

    def __repr__(self) -> str:
        return (
            f"acc={self.accuracy:.3f} macro_f1={self.macro_f1:.3f} "
            f"rare{self.rare_class_labels}: "
            f"F1={self.rare_class_f1:.3f} "
            f"P={self.rare_class_precision:.3f} "
            f"R={self.rare_class_recall:.3f}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list,
    rare_class_labels: list,
) -> ClassificationMetrics:
    """Compute overall + rare-class metrics.

    Args:
        y_true: (n,) integer array of ground-truth class indices.
        y_pred: (n,) integer array of predicted class indices.
        class_labels: full ordered list of class indices present in the
            task (e.g. `[0, 1, 2, 3, 4]`), used for macro averaging and
            per-class breakdown so a class absent from one batch doesn't
            silently disappear from the report.
        rare_class_labels: which of `class_labels` are "rare", typically
            the `.rare_classes` output of
            `src.rare_class.entropy.identify_rare_classes` — kept as an
            explicit argument (not recomputed here) so metrics always use
            whichever rare-class definition the experiment actually used.

    Returns:
        `ClassificationMetrics`. If none of `rare_class_labels` appear in
        `y_true`, rare-class F1/precision/recall are reported as 0.0 with
        a note rather than raising, since an empty test-domain slice for
        the rare class is itself a meaningful (bad) result worth logging.

    Raises:
        ValueError: if `rare_class_labels` is empty, or contains a label
            not present in `class_labels`.
    """
    if not rare_class_labels:
        raise ValueError("rare_class_labels must be non-empty.")
    unknown = set(rare_class_labels) - set(class_labels)
    if unknown:
        raise ValueError(f"rare_class_labels {unknown} not in class_labels {class_labels}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    per_class_f1 = dict(zip(class_labels, per_class.tolist()))

    rare_f1 = f1_score(y_true, y_pred, labels=rare_class_labels, average="macro", zero_division=0)
    rare_p = precision_score(y_true, y_pred, labels=rare_class_labels, average="macro", zero_division=0)
    rare_r = recall_score(y_true, y_pred, labels=rare_class_labels, average="macro", zero_division=0)

    return ClassificationMetrics(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        rare_class_labels=list(rare_class_labels),
        rare_class_f1=float(rare_f1),
        rare_class_precision=float(rare_p),
        rare_class_recall=float(rare_r),
        per_class_f1=per_class_f1,
    )


def modality_degradation(
    full_metrics: ClassificationMetrics, degraded_metrics: ClassificationMetrics
) -> dict:
    """Section 6.3's "missing-modality degradation" metric: how much
    rare-class F1 drops when a modality is removed at test time.

    Args:
        full_metrics: metrics with all modalities present.
        degraded_metrics: metrics with one modality missing/zeroed.

    Returns:
        Dict with absolute and relative drop in rare-class F1. Positive
        `absolute_drop` means performance got worse without the modality
        (the expected direction); a negative value would mean the model
        does BETTER without that modality, which is worth flagging, not
        hiding, in a results table.
    """
    absolute_drop = full_metrics.rare_class_f1 - degraded_metrics.rare_class_f1
    relative_drop = (
        absolute_drop / full_metrics.rare_class_f1 if full_metrics.rare_class_f1 > 0 else float("nan")
    )
    return {
        "full_rare_f1": full_metrics.rare_class_f1,
        "degraded_rare_f1": degraded_metrics.rare_class_f1,
        "absolute_drop": absolute_drop,
        "relative_drop": relative_drop,
    }
