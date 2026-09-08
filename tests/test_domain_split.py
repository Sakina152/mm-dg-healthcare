"""Unit tests for src/data/domain_split.py. Synthetic dataframe only."""
import pandas as pd
import pytest

from src.data.domain_split import all_pairwise_domain_splits, split_by_domain


def _toy_df():
    return pd.DataFrame(
        {
            "id": range(12),
            "site": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "label": ["NORM", "MI", "NORM", "MI"] * 3,
        }
    )


class TestSplitByDomain:
    def test_basic_split_is_disjoint_and_complete_for_named_domains(self):
        df = _toy_df()
        split = split_by_domain(df, "site", train_domains=[1], test_domains=[2])
        assert set(split.train_df["site"]) == {1}
        assert set(split.test_df["site"]) == {2}
        assert len(split.train_df) == 4
        assert len(split.test_df) == 4

    def test_multiple_train_domains(self):
        df = _toy_df()
        split = split_by_domain(df, "site", train_domains=[1, 2], test_domains=[3])
        assert len(split.train_df) == 8
        assert len(split.test_df) == 4

    def test_missing_domain_column_raises(self):
        df = _toy_df()
        with pytest.raises(ValueError, match="not found"):
            split_by_domain(df, "hospital", [1], [2])

    def test_overlapping_domains_raises(self):
        df = _toy_df()
        with pytest.raises(ValueError, match="overlap"):
            split_by_domain(df, "site", train_domains=[1, 2], test_domains=[2, 3])

    def test_empty_train_split_raises(self):
        df = _toy_df()
        with pytest.raises(ValueError, match="No rows found"):
            split_by_domain(df, "site", train_domains=[999], test_domains=[1])

    def test_empty_test_split_raises(self):
        df = _toy_df()
        with pytest.raises(ValueError, match="No rows found"):
            split_by_domain(df, "site", train_domains=[1], test_domains=[999])

    def test_summary_string_contains_domain_info(self):
        df = _toy_df()
        split = split_by_domain(df, "site", [1], [2])
        s = split.summary()
        assert "site" in s and "[1]" in s and "[2]" in s


class TestAllPairwiseDomainSplits:
    def test_generates_every_ordered_pair_excluding_self_pairs(self):
        df = _toy_df()
        splits = all_pairwise_domain_splits(df, "site")
        pairs = {(s.train_domains[0], s.test_domains[0]) for s in splits}
        expected = {(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)}
        assert pairs == expected

    def test_no_split_has_overlapping_train_test(self):
        df = _toy_df()
        for s in all_pairwise_domain_splits(df, "site"):
            assert set(s.train_domains).isdisjoint(set(s.test_domains))
