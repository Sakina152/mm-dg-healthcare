"""Domain-shift splitting.

The reference paper trains on one hospital ("Center A") and tests, unmodified,
on a different hospital ("Center B") — that mismatch is the whole point of a
domain-generalization experiment. A random train/test split would put
patients from every site in both train and test, which hides the exact
problem this project studies. So splitting must be done by a domain key
(here, PTB-XL's recording `site`), never by `sklearn.train_test_split`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DomainSplit:
    """Container for one domain-shift split.

    Attributes:
        train_df: rows whose domain key is in `train_domains`.
        test_df: rows whose domain key is in `test_domains`.
        train_domains: the domain id(s) used for training.
        test_domains: the domain id(s) held out for testing.
        domain_col: which column was used as the domain key.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_domains: list
    test_domains: list
    domain_col: str

    def summary(self) -> str:
        return (
            f"Domain split on '{self.domain_col}': "
            f"train={self.train_domains} (n={len(self.train_df)}), "
            f"test={self.test_domains} (n={len(self.test_df)})"
        )


def split_by_domain(
    df: pd.DataFrame,
    domain_col: str,
    train_domains: list,
    test_domains: list,
) -> DomainSplit:
    """Split a dataframe by domain id, e.g. PTB-XL's `site` column.

    Args:
        df: metadata dataframe, one row per recording/patient.
        domain_col: name of the column identifying the domain (e.g. `"site"`
            for PTB-XL, or `"hospital"` / `"scanner"` for another dataset).
        train_domains: list of domain values to use for training.
        test_domains: list of domain values to hold out for testing. Must
            not overlap with `train_domains` — this is checked and will
            raise, because overlapping domains silently turns a
            domain-generalization experiment back into an ordinary
            same-distribution split.

    Returns:
        A `DomainSplit` with `train_df` and `test_df` populated.

    Raises:
        ValueError: if `domain_col` is missing, if `train_domains` and
            `test_domains` overlap, or if either resulting split is empty.
    """
    if domain_col not in df.columns:
        raise ValueError(
            f"'{domain_col}' not found in dataframe columns: {list(df.columns)}"
        )

    overlap = set(train_domains) & set(test_domains)
    if overlap:
        raise ValueError(
            f"train_domains and test_domains overlap on {overlap} — a "
            "domain-generalization split must not share domains between "
            "train and test."
        )

    train_df = df[df[domain_col].isin(train_domains)].reset_index(drop=True)
    test_df = df[df[domain_col].isin(test_domains)].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(f"No rows found for train_domains={train_domains}")
    if len(test_df) == 0:
        raise ValueError(f"No rows found for test_domains={test_domains}")

    return DomainSplit(
        train_df=train_df,
        test_df=test_df,
        train_domains=list(train_domains),
        test_domains=list(test_domains),
        domain_col=domain_col,
    )


def all_pairwise_domain_splits(
    df: pd.DataFrame, domain_col: str
) -> list[DomainSplit]:
    """Build every (train-on-one-domain, test-on-another) pair, matching the
    reference paper's "across-trial validation" protocol (Table 2: Train A
    Test B, Train B Test A). Useful for a full ablation sweep across all
    sites rather than picking just one direction by hand.

    Args:
        df: metadata dataframe.
        domain_col: domain id column.

    Returns:
        List of `DomainSplit`, one per ordered pair of distinct domains.
    """
    domains = sorted(df[domain_col].dropna().unique().tolist())
    splits = []
    for train_d in domains:
        for test_d in domains:
            if train_d == test_d:
                continue
            splits.append(
                split_by_domain(df, domain_col, [train_d], [test_d])
            )
    return splits
