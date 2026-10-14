"""PTB-XL dataset wrapper for ECG plus non-diagnostic patient metadata."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.ptbxl_dataset import PTBXLDataset


class PTBXLMetadataDataset(Dataset):
    """Adds age/sex features normalized using source-training statistics only.

    ``metadata_stats`` must be computed from the training split and reused for
    validation/target data. SCP codes, superclass descriptions and report text
    are never included as features.
    """
    FEATURE_COLUMNS = ("age", "sex")

    def __init__(self, metadata: pd.DataFrame, ptbxl_dir, label_to_idx=None,
                 metadata_stats: dict | None = None, signal_length: int = 1000):
        self.base = PTBXLDataset(metadata, ptbxl_dir, label_to_idx, signal_length)
        self.metadata = metadata.reset_index(drop=True).copy()
        numeric = self.metadata.loc[:, list(self.FEATURE_COLUMNS)].copy()
        # PTB-XL commonly stores sex numerically, but accept string variants too.
        if "sex" in numeric.columns:
            sex = numeric["sex"]
            sex_text = sex.astype(str).str.strip().str.lower()
            sex_map = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}
            mapped = sex_text.map(sex_map)
            numeric["sex"] = pd.to_numeric(sex, errors="coerce").where(
                pd.to_numeric(sex, errors="coerce").notna(), mapped
            )
        for col in numeric.columns:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        if metadata_stats is None:
            medians = numeric.median().fillna(0.0)
            filled = numeric.fillna(medians)
            means = filled.mean()
            stds = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
            self.metadata_stats = {
                "median": medians.to_dict(), "mean": means.to_dict(), "std": stds.to_dict()
            }
        else:
            self.metadata_stats = metadata_stats
        medians = pd.Series(self.metadata_stats["median"])
        means = pd.Series(self.metadata_stats["mean"])
        stds = pd.Series(self.metadata_stats["std"]).replace(0, 1.0)
        filled = numeric.fillna(medians).fillna(0.0)
        standardized = (filled - means) / stds
        self.features = np.nan_to_num(standardized.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["metadata"] = torch.from_numpy(self.features[idx])
        return item
