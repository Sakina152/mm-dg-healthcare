"""Unit tests for src/rare_class/entropy.py.

Runs entirely on synthetic NumPy arrays — no dataset download needed, so
these can (and should) run in CI or before any real data is available.
"""
import numpy as np
import pytest

from src.rare_class.entropy import class_entropy, identify_rare_classes


def _make_tight_cluster(n, dim, center, scale=0.01, seed=0):
    """A class whose points are nearly identical -> low entropy expected."""
    rng = np.random.default_rng(seed)
    return center + rng.normal(scale=scale, size=(n, dim))


def _make_spread_cluster(n, dim, center, scale=5.0, seed=1):
    """A class whose points are widely scattered -> higher entropy expected."""
    rng = np.random.default_rng(seed)
    return center + rng.normal(scale=scale, size=(n, dim))


class TestClassEntropy:
    def test_tight_cluster_has_low_entropy_relative_to_uniform_bound(self):
        emb = _make_tight_cluster(n=50, dim=8, center=np.zeros(8))
        theta = class_entropy(emb, k=5)
        # Entropy is bounded above by log2(n) (uniform density case).
        assert 0.0 <= theta <= np.log2(50) + 1e-6

    def test_single_point_class_returns_zero_not_nan(self):
        emb = np.array([[1.0, 2.0, 3.0]])
        theta = class_entropy(emb, k=5)
        assert theta == 0.0

    def test_k_larger_than_available_neighbors_does_not_crash(self):
        emb = _make_tight_cluster(n=3, dim=4, center=np.ones(4))
        # k=50 requested but only 2 other points exist per point
        theta = class_entropy(emb, k=50)
        assert np.isfinite(theta)


class TestIdentifyRareClasses:
    def test_flags_a_genuinely_sparse_outlier_class(self):
        # Three tight, well-separated, evenly sized clusters ("common"
        # classes) and one small scattered cluster ("rare" class) far away.
        rng = np.random.default_rng(42)
        common_a = _make_tight_cluster(80, 6, center=np.array([0, 0, 0, 0, 0, 0]), seed=1)
        common_b = _make_tight_cluster(80, 6, center=np.array([10, 0, 0, 0, 0, 0]), seed=2)
        common_c = _make_tight_cluster(80, 6, center=np.array([0, 10, 0, 0, 0, 0]), seed=3)
        rare = _make_spread_cluster(8, 6, center=np.array([50, 50, 0, 0, 0, 0]), scale=8.0, seed=4)

        embeddings = np.vstack([common_a, common_b, common_c, rare])
        labels = np.array(
            ["A"] * 80 + ["B"] * 80 + ["C"] * 80 + ["RARE"] * 8
        )

        result = identify_rare_classes(embeddings, labels, k=5)

        assert "RARE" in result.rare_classes
        assert set(result.class_entropy.keys()) == {"A", "B", "C", "RARE"}

    def test_raises_on_mismatched_lengths(self):
        embeddings = np.zeros((10, 4))
        labels = np.zeros(9)
        with pytest.raises(ValueError, match="same length"):
            identify_rare_classes(embeddings, labels)

    def test_raises_on_single_class(self):
        embeddings = np.zeros((10, 4))
        labels = np.array(["A"] * 10)
        with pytest.raises(ValueError, match="at least 2 classes"):
            identify_rare_classes(embeddings, labels)

    def test_is_rare_helper_matches_rare_classes_list(self):
        embeddings = np.vstack(
            [
                _make_tight_cluster(50, 4, center=np.zeros(4), seed=1),
                _make_spread_cluster(5, 4, center=np.array([30, 30, 0, 0]), scale=6.0, seed=2),
            ]
        )
        labels = np.array(["COMMON"] * 50 + ["RARE"] * 5)
        result = identify_rare_classes(embeddings, labels, k=4)
        for lbl in result.class_entropy:
            assert result.is_rare(lbl) == (lbl in result.rare_classes)
