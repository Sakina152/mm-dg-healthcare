# Robust Multimodal Fusion for Domain Generalization in Rare-Class Medical Detection

Companion codebase for the project proposal *"Robust Multimodal Fusion for Domain
Generalization in Rare-Class Medical Detection Under Missing-Modality Conditions."*
Built incrementally, phase by phase, following the proposal's methodology
(Section 6). Every phase is runnable on its own — you don't need later phases
to reproduce earlier results.

## Build status

| Phase | Proposal section | Status |
|---|---|---|
| 0. Project scaffolding, config, seeding, tests | — | ✅ Done |
| 1. Baseline (signal-only) + domain-shift split | 6.2 steps 1 | ✅ Done |
| 2. Rare-class identification (entropy method) | 6.2 step 2 | ✅ Done |
| 3. Multimodal fusion model | 6.2 step 3 | 🚧 Scaffolded, not yet trained |
| 4. Domain generalization evaluation of fusion | 6.2 step 4 | 🚧 Scaffolded |
| 5. Missing-modality robustness training | 6.2 step 5 | 🚧 Scaffolded |
| 6. Full ablation suite | 6.2 step 6 | 🚧 Scaffolded |

"Scaffolded" means the folder, module, config, and notebook exist with a
clear interface and docstrings, so the next build increment plugs directly
into it — nothing about the architecture needs to be re-decided later.

## Why PTB-XL first

MIMIC-CXR (the proposal's other dataset) requires PhysioNet credentialing,
which can take days to be approved. **PTB-XL is fully public, no
credentialing needed**, and already has the two things this project needs:

- A **domain-shift axis that's already in the metadata**: the `site` field
  (5 recording sites) and `device` field, which we use exactly like the
  reference paper's Center A / Center B split.
- A **second, independently-authored modality**: each diagnostic SCP code is
  linked to a standardized clinical text statement (`scp_statements.csv`),
  which is genuinely independent of the waveform pixels/samples — it's not
  text extracted *from* the signal by a detector, which is the exact gap
  the proposal identifies (Gap A) in prior work (GenEval).

  **Honest limitation**: these are standardized coded statements, not
  free-text physician narratives like MIMIC-CXR reports. That's a weaker
  form of "independent text" than we'll get once MIMIC-CXR access is
  approved. The codebase is written so swapping in MIMIC-CXR later
  (`src/data/mimic_cxr_dataset.py`, currently a stub) doesn't require
  touching the model or training code — only the `Dataset` class changes.

## Folder structure

```
mm-dg-healthcare/
├── README.md                     ← you are here
├── requirements.txt
├── .gitignore
├── configs/                      ← one YAML per experiment, nothing hardcoded in code
│   ├── baseline_signal_only.yaml
│   ├── fusion_full.yaml          (scaffold)
│   └── fusion_robust.yaml        (scaffold)
├── data/
│   ├── raw/                      ← downloaded PTB-XL goes here (gitignored)
│   └── processed/                ← cached splits / features (gitignored)
├── src/
│   ├── data/
│   │   ├── ptbxl_dataset.py      ← PyTorch Dataset for ECG waveforms + labels
│   │   ├── ptbxl_text.py         ← loads the independent SCP text-statement modality
│   │   ├── mimic_cxr_dataset.py  (scaffold — for once PhysioNet access is granted)
│   │   └── domain_split.py       ← train-on-site-X / test-on-site-Y splitting
│   ├── models/
│   │   ├── signal_encoder.py     ← 1D-CNN ECG encoder
│   │   ├── text_encoder.py       (scaffold — Phase 3)
│   │   ├── fusion.py             (scaffold — Phase 3)
│   │   └── classifier.py         ← classification head
│   ├── rare_class/
│   │   └── entropy.py            ← Definitions 1–2 from the reference paper
│   ├── training/
│   │   ├── trainer.py            ← generic train/eval loop, checkpointing
│   │   └── modality_dropout.py   (scaffold — Phase 5)
│   ├── evaluation/
│   │   └── metrics.py            ← rare-class F1/precision/recall, overall accuracy
│   └── utils/
│       ├── seed.py               ← one function, sets every RNG
│       └── config.py             ← YAML → dict/Namespace loader
├── notebooks/                    ← Colab-first; each one clones/pulls this repo
│   ├── 00_colab_setup.ipynb
│   ├── 01_data_exploration_ptbxl.ipynb
│   ├── 02_baseline_signal_only.ipynb
│   ├── 03_rare_class_identification.ipynb
│   ├── 04_multimodal_fusion.ipynb        (scaffold)
│   ├── 05_missing_modality_robustness.ipynb (scaffold)
│   └── 06_ablation_results.ipynb         (scaffold)
├── scripts/
│   └── download_ptbxl.sh         ← one-line PTB-XL download (no login required)
├── experiments/results/          ← metrics.json / checkpoints land here (gitignored)
└── tests/
    ├── test_entropy.py           ← unit tests for the rare-class math, synthetic data
    └── test_domain_split.py      ← unit tests for the site-based split, synthetic data
```

## Reproducibility rules this codebase follows

1. **Nothing is hardcoded.** Every experiment is one YAML file in `configs/`
   plus a command. Change the config, not the code, to run a variant.
2. **One seed function, called once, at the top of every entry point**
   (`src/utils/seed.py`) — seeds Python, NumPy, and PyTorch (CPU + CUDA).
3. **Every run writes its exact config next to its results** in
   `experiments/results/<run_name>/`, so a result can never get separated
   from the settings that produced it.
4. **Splits are computed by site ID, not randomly**, for domain-shift
   experiments — random splits would leak the same hospital into train and
   test and silently hide the domain-shift problem this project is about.
5. **Unit tests don't need the dataset.** `tests/` runs against small
   synthetic arrays so you (or a grader) can verify the core math
   (rare-class entropy, domain split logic) in seconds, with no download.

## Quickstart (local)

```bash
git clone <your-repo-url> mm-dg-healthcare
cd mm-dg-healthcare
pip install -r requirements.txt
bash scripts/download_ptbxl.sh        # ~1.7 GB, no login needed
pytest tests/ -v                       # sanity check, no data needed for this
python -m src.training.trainer --config configs/baseline_signal_only.yaml
```

## Quickstart (Google Colab)

Open `notebooks/00_colab_setup.ipynb` first — it clones this repo, installs
`requirements.txt`, and downloads PTB-XL straight into Colab's disk (or
optionally caches it to your Google Drive so you don't re-download every
session). Every later notebook assumes 00 has been run once in that Colab
runtime.

## Current results (Phase 1 — baseline, no fusion yet)

Populated automatically by `02_baseline_signal_only.ipynb` /
`src/training/trainer.py` into `experiments/results/baseline_signal_only/metrics.json`.
Table is intentionally empty until you run it — no fabricated numbers here.

| Train site → Test site | Accuracy | Macro-F1 | Rare-class F1 |
|---|---|---|---|
| _run pending_ | | | |
