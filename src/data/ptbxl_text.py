"""The independent text modality for PTB-XL — used from Phase 3 onward.

This is the piece that directly targets Gap A from the proposal: prior
multimodal work (GenEval) generates its "text knowledge" FROM the same
image via an object detector, so it isn't really a second, independently
collected source. Here, the text comes from `scp_statements.csv`'s
`SCP-ECG Statement Description` field — a standardized clinical
description authored independently of any specific patient's waveform,
looked up by diagnostic code, not derived from the signal pixels/samples.

Honest scope note (see README): these are standardized coded statements,
not free-text physician narratives per patient like MIMIC-CXR reports.
That's a real limitation of doing Phase 3 on PTB-XL rather than MIMIC-CXR.
The `PTBXLTextLookup` interface below is intentionally identical in shape
to what `mimic_cxr_dataset.py`'s eventual text loader will expose
(`get_text(id) -> str`), so swapping datasets later doesn't require
touching `src/models/text_encoder.py` or the fusion code.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


class PTBXLTextLookup:
    """Maps a PTB-XL record's `scp_codes` to a short independent text
    description, built by joining each code's statement descriptions.
    """

    def __init__(self, ptbxl_dir: str | Path):
        """
        Args:
            ptbxl_dir: root PTB-XL directory containing `scp_statements.csv`.

        Raises:
            FileNotFoundError: if `scp_statements.csv` is missing.
        """
        ptbxl_dir = Path(ptbxl_dir)
        scp_path = ptbxl_dir / "scp_statements.csv"
        if not scp_path.exists():
            raise FileNotFoundError(
                f"{scp_path} not found. Run scripts/download_ptbxl.sh first."
            )
        scp_df = pd.read_csv(scp_path, index_col=0)
        # SCP-ECG Statement Description is the human-readable clinical text
        # for each code, e.g. "myocardial infarction", "left ventricular
        # hypertrophy" — independently authored, not derived from any one
        # patient's waveform.
        self._code_to_text = scp_df["SCP-ECG Statement Description"].fillna(
            scp_df.index.to_series()
        ).to_dict()

    def get_text(self, scp_codes) -> str:
        """Build one text string for a record from its `scp_codes`.

        Args:
            scp_codes: either the raw string from `ptbxl_database.csv`
                (e.g. `"{'NORM': 100.0}"`) or an already-parsed dict.

        Returns:
            A single string joining the statement descriptions for every
            code present, highest-confidence first — e.g.
            "normal ECG; sinus rhythm". Falls back to the raw code if no
            description is known for it, rather than silently dropping it.
        """
        if isinstance(scp_codes, str):
            scp_codes = ast.literal_eval(scp_codes)
        ordered_codes = sorted(scp_codes, key=scp_codes.get, reverse=True)
        parts = [self._code_to_text.get(c, c) for c in ordered_codes]
        return "; ".join(parts)
