"""PTB-XL waveform dataset.

PTB-XL (Wagner et al., PhysioNet, fully public — no credentialing) ships
12-lead ECG records plus a metadata CSV (`ptbxl_database.csv`) with, among
other things:

  - `site`            : recording site, 1-5  -> our domain-shift axis,
                         used exactly like the reference paper's
                         Center A / Center B.
  - `scp_codes`        : dict of diagnostic codes -> confidence, which we
                         collapse to one of 5 diagnostic superclasses
                         (NORM, MI, STTC, CD, HYP) via
                         `scp_statements.csv`. HYP is the smallest class in
                         the full dataset and is a natural rare-class
                         candidate — but which class is actually "rare" is
                         decided empirically per split by
                         `src.rare_class.entropy`, not hardcoded here.
  - `filename_lr`       : path to the 100Hz waveform record (what we use;
                         PTB-XL also ships a 500Hz version if higher
                         resolution is ever needed).

Directory expected (produced by `scripts/download_ptbxl.sh`):
    data/raw/ptbxl/ptbxl_database.csv
    data/raw/ptbxl/scp_statements.csv
    data/raw/ptbxl/records100/...
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_ptbxl_metadata(ptbxl_dir: str | Path) -> pd.DataFrame:
    """Load and lightly process PTB-XL's metadata CSV: parses `scp_codes`
    from string to dict, and adds a `superclass` column (the single
    highest-confidence diagnostic superclass per record — a simplification;
    PTB-XL records can carry multiple codes, but a single label per record
    keeps the classification task and the entropy/rare-class computation
    well-defined for this project's scope).

    Args:
        ptbxl_dir: path to the extracted PTB-XL directory (contains
            `ptbxl_database.csv` and `scp_statements.csv`).

    Returns:
        DataFrame with one row per record, including `site`, `superclass`,
        and `filename_lr` columns ready for `PTBXLDataset`.

    Raises:
        FileNotFoundError: if the expected CSVs aren't present — usually
            means `scripts/download_ptbxl.sh` hasn't been run yet.
    """
    ptbxl_dir = Path(ptbxl_dir)
    db_path = ptbxl_dir / "ptbxl_database.csv"
    scp_path = ptbxl_dir / "scp_statements.csv"
    for p in (db_path, scp_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run scripts/download_ptbxl.sh first, or "
                "pass the correct --ptbxl_dir."
            )

    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_df = scp_df[scp_df["diagnostic"] == 1]
    code_to_superclass = scp_df["diagnostic_class"].to_dict()

    def pick_superclass(codes: dict) -> str | None:
        # Among this record's codes, keep those that map to a known
        # diagnostic superclass, then take the one with highest reported
        # confidence. Records with no diagnostic code are dropped by the
        # caller (superclass will be None).
        candidates = {
            code_to_superclass[c]: conf
            for c, conf in codes.items()
            if c in code_to_superclass
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    df["superclass"] = df["scp_codes"].apply(pick_superclass)
    df = df.dropna(subset=["superclass", "site"]).copy()
    df["site"] = df["site"].astype(int)
    return df.reset_index()


class PTBXLDataset(Dataset):
    """Loads 100Hz 12-lead waveforms lazily (one `wfdb.rdsamp` call per
    `__getitem__`) so we never hold the full signal set in memory —
    PTB-XL is ~21k records, small enough to preload, but this keeps the
    same class usable on larger future datasets without a rewrite.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        ptbxl_dir: str | Path,
        label_to_idx: dict[str, int] | None = None,
        signal_length: int = 1000,  # 10s @ 100Hz
    ):
        """
        Args:
            metadata: output of `load_ptbxl_metadata`, already filtered to
                the desired split (e.g. via `domain_split.split_by_domain`).
            ptbxl_dir: root PTB-XL directory (same one passed to
                `load_ptbxl_metadata`).
            label_to_idx: mapping from superclass string to integer index.
                If None, built from `DIAGNOSTIC_SUPERCLASSES` so label
                indices are stable across train/test splits (built once,
                shared, never re-derived per-split — a per-split mapping
                would silently misalign train and test label indices).
            signal_length: samples to keep per record; PTB-XL's 100Hz
                records are 1000 samples (10s). Shorter/longer records are
                center-cropped/zero-padded to this length.
        """
        self.metadata = metadata.reset_index(drop=True)
        self.ptbxl_dir = Path(ptbxl_dir)
        self.label_to_idx = label_to_idx or {
            c: i for i, c in enumerate(DIAGNOSTIC_SUPERCLASSES)
        }
        self.signal_length = signal_length

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_signal(self, filename_lr: str) -> np.ndarray:
        record_path = self.ptbxl_dir / filename_lr
        signal, _ = wfdb.rdsamp(str(record_path))  # (samples, 12 leads)
        signal = signal.T  # -> (12, samples), channel-first for Conv1d
        n = signal.shape[1]
        if n == self.signal_length:
            return signal
        if n > self.signal_length:
            start = (n - self.signal_length) // 2
            return signal[:, start : start + self.signal_length]
        pad = self.signal_length - n
        return np.pad(signal, ((0, 0), (0, pad)))

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        signal = self._load_signal(row["filename_lr"])
        signal = np.nan_to_num(signal, nan=0.0).astype(np.float32)

        return {
            "signal": torch.from_numpy(signal),  # (12, signal_length)
            "label": torch.tensor(self.label_to_idx[row["superclass"]], dtype=torch.long),
            "site": int(row["site"]),
            "ecg_id": int(row["ecg_id"]),
        }
