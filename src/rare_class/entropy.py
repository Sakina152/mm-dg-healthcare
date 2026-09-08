"""Rare-class identification via class-wise entropy.

Implements Definitions 1-2 and Equations 1-3 from the reference paper
(Banerjee, Srivathsan & Gupta, 2026, "Experience with Single Domain
Generalization in Real World Medical Imaging Deployments"):

    density        λ(x_i)  = mean inverse distance to x_i's within-class
                              k-nearest-neighbors                    (Eq. 1)
    class density  γ(x_i)  = λ(x_i) normalized within its class       (Eq. 2)
    class entropy  θ_c     = -sum_i γ(x_i) * log2(γ(x_i))             (Eq. 3)

A class is "rare" (Definition 2) if its entropy falls more than one
standard deviation outside the mean entropy across all classes:

    θ_r > θ_mean + σ_θ   or   θ_r < θ_mean - σ_θ

We use this instead of just "whichever class has fewest examples" because
the reference paper's whole point (and the reason a rare class is *hard*,
not just small) is that its embeddings are also spread out / inconsistent
— that's what entropy captures and a raw class count doesn't.

This module is deliberately independent of any specific model or dataset:
it takes an (embeddings, labels) pair, from any encoder, and returns which
label(s) qualify as rare. Phase 1 calls it on the baseline signal encoder's
embeddings; Phase 3+ can call it again on the fused embeddings to check
whether fusion changes which class looks "rare."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class EntropyResult:
    """Per-class entropy plus which classes were flagged as rare.

    Attributes:
        class_entropy: {class_label: theta} for every class present.
        mean_entropy: mean of `class_entropy` values (theta_M in the paper).
        std_entropy: population std of `class_entropy` values (sigma_theta).
        rare_classes: labels whose entropy falls outside
            [mean - std, mean + std], per Definition 2.
    """

    class_entropy: dict
    mean_entropy: float
    std_entropy: float
    rare_classes: list

    def is_rare(self, label) -> bool:
        return label in self.rare_classes


def _within_class_density(
    embeddings: np.ndarray, k: int, eps: float = 1e-8
) -> np.ndarray:
    """Equation 1: lambda(x_i) for every point in `embeddings`, where the
    k-nearest-neighbor search is restricted to `embeddings` itself (i.e.
    call this once per class, with only that class's embeddings passed in).

    Args:
        embeddings: (n_samples, dim) array, all belonging to one class.
        k: number of nearest neighbors to average over. Capped at
            n_samples - 1 automatically (needed for very small rare classes
            where k could otherwise exceed the available neighbor count).
        eps: added to distances before inverting, to avoid division by
            zero for coincident points.

    Returns:
        (n_samples,) array of density values, one per input point.
    """
    n = embeddings.shape[0]
    k_eff = max(1, min(k, n - 1))
    if n == 1:
        # A "class" with a single observation has no neighbors; density is
        # undefined by the formula. We return 0 so it doesn't silently
        # explode the entropy computation, and let the caller decide
        # whether n=1 classes are meaningful in their dataset.
        return np.zeros(1)

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(embeddings)  # +1: includes self
    distances, _ = nn.kneighbors(embeddings)
    distances = distances[:, 1:]  # drop the self-match at distance 0
    inv_dist = 1.0 / (distances + eps)
    return inv_dist.mean(axis=1)


def class_entropy(embeddings: np.ndarray, k: int = 5) -> float:
    """Equations 1-3 combined, for a single class's embeddings.

    Args:
        embeddings: (n_samples, dim) array for one class only.
        k: neighbors used in the density estimate (Eq. 1).

    Returns:
        Scalar entropy theta for this class.
    """
    lam = _within_class_density(embeddings, k=k)
    total = lam.sum()
    if total <= 0:
        return 0.0
    gamma = lam / total  # Eq. 2: normalize density within the class
    # Eq. 3: -sum(gamma * log2(gamma)), guarding gamma=0 (0 log 0 := 0)
    nonzero = gamma[gamma > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def identify_rare_classes(
    embeddings: np.ndarray, labels: np.ndarray, k: int = 5
) -> EntropyResult:
    """Definition 2: compute per-class entropy and flag rare classes.

    Args:
        embeddings: (n_samples, dim) array across ALL classes.
        labels: (n_samples,) array of class labels, aligned with `embeddings`.
        k: neighbors used in the within-class density estimate.

    Returns:
        `EntropyResult` with per-class entropy, the mean/std used as the
        Definition-2 threshold, and the resulting list of rare classes.

    Raises:
        ValueError: if `embeddings` and `labels` have mismatched lengths,
            or if fewer than 2 classes are present (entropy comparison
            across classes is meaningless with only one class).
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    if len(embeddings) != len(labels):
        raise ValueError(
            f"embeddings ({len(embeddings)}) and labels ({len(labels)}) "
            "must have the same length."
        )

    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) < 2:
        raise ValueError(
            "Need at least 2 classes to identify a rare class relative to "
            f"the others; got {len(unique_labels)}."
        )

    theta = {}
    for lbl in unique_labels:
        mask = labels == lbl
        theta[lbl] = class_entropy(embeddings[mask], k=k)

    values = np.array(list(theta.values()))
    theta_mean = float(values.mean())
    theta_std = float(values.std(ddof=0))  # population std, matches paper's formula

    rare = [
        lbl
        for lbl, t in theta.items()
        if t > theta_mean + theta_std or t < theta_mean - theta_std
    ]

    return EntropyResult(
        class_entropy=theta,
        mean_entropy=theta_mean,
        std_entropy=theta_std,
        rare_classes=rare,
    )
