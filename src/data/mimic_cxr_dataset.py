"""MIMIC-CXR dataset — SCAFFOLD, not yet implemented.

Status: blocked on PhysioNet credentialed access (requires a completed
CITI training certificate + data use agreement; typically takes a few
days for approval). Do not attempt to download or hardcode any MIMIC-CXR
data path until that access is confirmed.

Why this file exists already, unimplemented: the proposal's Section 6.1
lists MIMIC-CXR specifically because it has genuine free-text radiology
reports (independent of the image, written by a radiologist) — a stronger
version of the "independent text modality" than PTB-XL's coded statements
(see `ptbxl_text.py`'s docstring). Once access is granted, implementing
this module should be the ONLY change needed to run every Phase 3+
experiment on MIMIC-CXR instead of PTB-XL, because it will expose the same
two interfaces PTB-XL's loaders already expose:

    MIMICCXRDataset(metadata, ...)   — same shape as PTBXLDataset:
        __getitem__ returns {"image": Tensor, "label": Tensor, "site"/
        "domain_key": ..., "id": ...}

    MIMICCXRTextLookup.get_text(id) -> str   — same shape as
        PTBXLTextLookup.get_text(scp_codes) -> str

`src/models/fusion.py` and `src/training/trainer.py` are written against
these two interfaces, not against PTB-XL specifically, so they should not
need to change when this module is filled in.

Planned domain-shift axis for MIMIC-CXR (to implement once data is
available): split by acquisition metadata such as `ViewPosition` or care
unit, analogous to PTB-XL's `site` — needs to be finalized once the exact
fields available post-credentialing are confirmed.

TODO (Phase 3, once PhysioNet access is granted):
  1. Implement `load_mimic_cxr_metadata()` mirroring
     `ptbxl_dataset.load_ptbxl_metadata()`.
  2. Implement `MIMICCXRDataset` mirroring `PTBXLDataset`, loading JPEGs
     via `torchvision.io.read_image` instead of `wfdb.rdsamp`.
  3. Implement `MIMICCXRTextLookup` reading the paired `.txt` report files.
  4. Add `configs/dataset_mimic_cxr.yaml`.
"""

raise NotImplementedError(
    "MIMIC-CXR support is scaffolded but not implemented — requires "
    "PhysioNet credentialed access. See this module's docstring for the "
    "implementation plan. Use PTB-XL (src/data/ptbxl_dataset.py) for now."
)
