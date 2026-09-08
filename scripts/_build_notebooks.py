"""Internal build script — generates notebooks/*.ipynb from Python.

Not part of the shipped project's runtime; run once by the maintainer
whenever a notebook's content changes, so notebook source is reviewable
as plain Python (easier to diff/review) rather than hand-edited JSON.

Usage: python scripts/_build_notebooks.py
"""
import nbformat as nbf

NB_DIR = "notebooks"


def make_notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "colab": {"provenance": [], "name": "notebook.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------------------
# 00_colab_setup.ipynb
# ---------------------------------------------------------------------------
nb00 = make_notebook([
    md(
"""# 00 — Colab Setup

Run this notebook **first**, once per Colab runtime. It:
1. Clones this repo (or pulls latest if already cloned)
2. Installs `requirements.txt`
3. Downloads PTB-XL (public, no login) — optionally caches it to your
   Google Drive so you don't re-download every session
4. Runs the unit tests as a sanity check

Every other notebook in `notebooks/` assumes this has been run in the
current runtime."""
    ),
    md("## 1. Clone the repo"),
    code(
"""import os

REPO_URL = "https://github.com/<your-username>/mm-dg-healthcare.git"  # TODO: replace with your repo URL
REPO_DIR = "/content/mm-dg-healthcare"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    print("Repo already present, pulling latest...")
    !cd {REPO_DIR} && git pull

%cd {REPO_DIR}"""
    ),
    md(
"""## 2. Install dependencies

Colab already has `torch` preinstalled and compatible with GPU runtimes —
we skip reinstalling it here to avoid a slow, unnecessary CUDA-version
mismatch reinstall. Everything else in `requirements.txt` installs
normally."""
    ),
    code(
"""!pip install -q pandas scikit-learn scipy PyYAML tqdm wfdb transformers pytest
print("Dependencies installed.")"""
    ),
    md(
"""## 3. (Optional but recommended) Mount Google Drive for data caching

PTB-XL is ~1.7 GB. Downloading it every fresh Colab session is slow.
Caching it on Drive means you download once and every future session just
copies from Drive instead. Skip this cell if you'd rather download fresh
each time (e.g. on a shared/temporary account)."""
    ),
    code(
"""USE_DRIVE_CACHE = True  # set False to skip Drive entirely

if USE_DRIVE_CACHE:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_CACHE_DIR = "/content/drive/MyDrive/mm-dg-healthcare-data/ptbxl"
    import os
    os.makedirs(DRIVE_CACHE_DIR, exist_ok=True)
    print(f"Drive cache dir: {DRIVE_CACHE_DIR}")"""
    ),
    md("## 4. Download PTB-XL (skips automatically if already present)"),
    code(
"""import os

LOCAL_PTBXL_DIR = "data/raw/ptbxl"

if USE_DRIVE_CACHE and os.path.exists(os.path.join(DRIVE_CACHE_DIR, "ptbxl_database.csv")):
    print("Found cached PTB-XL on Drive — copying into the repo's data/raw/ (fast)...")
    !mkdir -p {LOCAL_PTBXL_DIR}
    !cp -r {DRIVE_CACHE_DIR}/* {LOCAL_PTBXL_DIR}/
else:
    print("No cache found — downloading from PhysioNet (a few minutes)...")
    !bash scripts/download_ptbxl.sh
    if USE_DRIVE_CACHE:
        print("Caching to Drive for next time...")
        !cp -r {LOCAL_PTBXL_DIR}/* {DRIVE_CACHE_DIR}/

!ls {LOCAL_PTBXL_DIR} | head -5"""
    ),
    md("## 5. Sanity check: run the unit tests (no dataset needed for these)"),
    code("""!python -m pytest tests/ -v"""),
    md(
"""## 6. Quick data sanity check

Confirms the metadata loads and the domain-shift `site` column looks as
expected before moving on to `01_data_exploration_ptbxl.ipynb`."""
    ),
    code(
"""from src.data.ptbxl_dataset import load_ptbxl_metadata

metadata = load_ptbxl_metadata("data/raw/ptbxl")
print(f"Loaded {len(metadata)} records")
print("\\nRecords per site:")
print(metadata["site"].value_counts().sort_index())
print("\\nRecords per diagnostic superclass:")
print(metadata["superclass"].value_counts())"""
    ),
    md("Setup complete. Continue to `01_data_exploration_ptbxl.ipynb`."),
])

# ---------------------------------------------------------------------------
# 01_data_exploration_ptbxl.ipynb
# ---------------------------------------------------------------------------
nb01 = make_notebook([
    md(
"""# 01 — PTB-XL Data Exploration

Look at what we're actually working with before training anything:
site/domain distribution, class balance, and a first look at raw
waveforms — all in plain language, matching Section 6.1 of the proposal.

Assumes `00_colab_setup.ipynb` has already been run in this runtime."""
    ),
    code(
"""%cd /content/mm-dg-healthcare
import pandas as pd
import matplotlib.pyplot as plt
from src.data.ptbxl_dataset import load_ptbxl_metadata, DIAGNOSTIC_SUPERCLASSES

metadata = load_ptbxl_metadata("data/raw/ptbxl")
metadata.head()"""
    ),
    md(
"""## Domain-shift axis: recording site

This is the column we'll split on to simulate "train at one hospital,
test at another," the same idea as the reference paper's Center A /
Center B experiments."""
    ),
    code(
"""site_counts = metadata["site"].value_counts().sort_index()
site_counts.plot(kind="bar", title="Records per PTB-XL site (our domain-shift axis)")
plt.xlabel("Site ID")
plt.ylabel("Number of records")
plt.show()

print(site_counts)"""
    ),
    md(
"""## Class balance per site

Worth checking whether the class balance itself shifts across sites —
if one site has almost no examples of a class, that's a harder
domain-shift pair to learn from and a natural first "hard" experiment."""
    ),
    code(
"""pd.crosstab(metadata["site"], metadata["superclass"])"""
    ),
    code(
"""ct = pd.crosstab(metadata["site"], metadata["superclass"], normalize="index")
ct.plot(kind="bar", stacked=True, figsize=(8, 5), title="Class proportions per site")
plt.ylabel("Proportion")
plt.legend(title="Superclass", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()"""
    ),
    md(
"""## A first look at a raw waveform

Sanity-check that a record actually loads and looks like an ECG before
we trust anything downstream."""
    ),
    code(
"""import wfdb
import numpy as np

row = metadata.iloc[0]
record_path = f"data/raw/ptbxl/{row['filename_lr']}"
signal, meta = wfdb.rdsamp(record_path)
print(f"Record: ecg_id={row['ecg_id']}, superclass={row['superclass']}, site={row['site']}")
print(f"Signal shape: {signal.shape} (samples, leads)")
print(f"Lead names: {meta['sig_name']}")

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for ax, lead_idx, lead_name in zip(axes, [0, 1, 6], [meta['sig_name'][0], meta['sig_name'][1], meta['sig_name'][6]]):
    ax.plot(signal[:, lead_idx])
    ax.set_ylabel(lead_name)
axes[-1].set_xlabel("Sample (100 Hz)")
fig.suptitle(f"ECG {row['ecg_id']} — {row['superclass']}")
plt.tight_layout()
plt.show()"""
    ),
    md(
"""## Independent text modality preview (Phase 3 groundwork)

Even though fusion isn't built yet, we can already look at the
independent text each record maps to — this is the "second modality"
Phase 3 will use, and it's worth eyeballing now to judge how informative
(or how repetitive) it looks before investing in a text encoder."""
    ),
    code(
"""from src.data.ptbxl_text import PTBXLTextLookup

text_lookup = PTBXLTextLookup("data/raw/ptbxl")
for _, row in metadata.sample(5, random_state=42).iterrows():
    text = text_lookup.get_text(row["scp_codes"])
    print(f"[{row['superclass']}] {text}")"""
    ),
    md("Continue to `02_baseline_signal_only.ipynb` to train the Phase 1 baseline."),
])

# ---------------------------------------------------------------------------
# 02_baseline_signal_only.ipynb
# ---------------------------------------------------------------------------
nb02 = make_notebook([
    md(
"""# 02 — Phase 1: Baseline Signal-Only Classifier Under Domain Shift

Trains the plain signal-only ECG classifier on one PTB-XL site and
evaluates it, unmodified, on a different site — reproducing the
"domain shift hurts performance" problem from the reference paper
(proposal Section 6.2, step 1), before any fusion is introduced.

Assumes `00_colab_setup.ipynb` has already been run in this runtime."""
    ),
    code(
"""%cd /content/mm-dg-healthcare
import torch
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))"""
    ),
    md(
"""## Run one domain pair from the command line

Uses `configs/baseline_signal_only.yaml` as-is (site 1 → site 2). This
calls the exact same `src/training/trainer.py` used outside Colab —
nothing Colab-specific about the actual training code, only this
notebook's setup cells are Colab-specific."""
    ),
    code(
"""!python -m src.training.trainer --config configs/baseline_signal_only.yaml"""
    ),
    md(
"""## Inspect results

Every run's metrics, full epoch history, and the exact config used are
written to `experiments/results/<run_dir>/` — never just printed and
lost."""
    ),
    code(
"""import json

run_dir = "experiments/results/baseline_signal_only_site1v2"
with open(f"{run_dir}/metrics.json") as f:
    metrics = json.load(f)
metrics"""
    ),
    code(
"""import pandas as pd

with open(f"{run_dir}/history.json") as f:
    history = json.load(f)

hist_df = pd.DataFrame([
    {"epoch": h["epoch"], "train_loss": h["train_loss"], "test_loss": h["test_loss"],
     "rare_class_f1": h["test_metrics"]["rare_class_f1"], "accuracy": h["test_metrics"]["accuracy"]}
    for h in history
])
hist_df.plot(x="epoch", y=["train_loss", "test_loss"], title="Loss curves")
plt.show()
hist_df.plot(x="epoch", y=["rare_class_f1", "accuracy"], title="Target-domain metrics per epoch")
plt.show()
hist_df.tail()"""
    ),
    md(
"""## Sweep every site pair (optional, matches the reference paper's full
across-trial validation table)

This trains one model per ordered site pair — with 5 PTB-XL sites that's
20 runs. On a Colab GPU each run is fast (small model, ~20 epochs), but
this cell will still take a while; run it when you actually want the full
table, not on every notebook re-run."""
    ),
    code(
"""RUN_FULL_SWEEP = False  # flip to True when you want the full table

if RUN_FULL_SWEEP:
    import yaml
    from src.data.ptbxl_dataset import load_ptbxl_metadata
    from src.data.domain_split import all_pairwise_domain_splits
    from src.utils.config import load_config, save_config
    from src.training.trainer import train as train_fn

    metadata = load_ptbxl_metadata("data/raw/ptbxl")
    splits = all_pairwise_domain_splits(metadata, "site")

    results = []
    base_cfg_dict = load_config("configs/baseline_signal_only.yaml").to_dict()

    for split in splits:
        cfg_dict = dict(base_cfg_dict)
        cfg_dict["data"] = dict(cfg_dict["data"])
        cfg_dict["data"]["train_sites"] = split.train_domains
        cfg_dict["data"]["test_sites"] = split.test_domains
        run_name = f"baseline_site{split.train_domains[0]}v{split.test_domains[0]}"
        cfg_dict["output"] = {"run_dir": f"experiments/results/{run_name}"}

        from src.utils.config import Config
        cfg = Config(cfg_dict)
        print(f"\\n=== {split.summary()} ===")
        final_metrics = train_fn(cfg)
        results.append({"train_site": split.train_domains[0], "test_site": split.test_domains[0], **final_metrics})

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv("experiments/results/baseline_sweep_summary.csv", index=False)
    sweep_df"""
    ),
    md("Continue to `03_rare_class_identification.ipynb` to formally identify the rare class per Definition 2."),
])

# ---------------------------------------------------------------------------
# 03_rare_class_identification.ipynb
# ---------------------------------------------------------------------------
nb03 = make_notebook([
    md(
"""# 03 — Phase 2: Rare-Class Identification (Entropy Method)

Applies the reference paper's Definitions 1–2 (implemented in
`src/rare_class/entropy.py`) to the **trained baseline model's own
embeddings**, so "which class is rare" is measured, not assumed — and
so we can later check whether fusion (Phase 3) changes which class looks
rare, which would itself be an interesting result.

Assumes `02_baseline_signal_only.ipynb` has been run (needs
`experiments/results/baseline_signal_only_site1v2/best_model.pt`)."""
    ),
    code(
"""%cd /content/mm-dg-healthcare
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.ptbxl_dataset import load_ptbxl_metadata, PTBXLDataset, DIAGNOSTIC_SUPERCLASSES
from src.data.domain_split import split_by_domain
from src.models.signal_encoder import SignalEncoder
from src.models.classifier import SignalOnlyClassifier
from src.rare_class.entropy import identify_rare_classes
from src.utils.config import load_config
from src.training.trainer import extract_embeddings
from torch.utils.data import DataLoader

cfg = load_config("configs/baseline_signal_only.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""
    ),
    md("## Load the trained baseline and rebuild its train-domain loader"),
    code(
"""metadata = load_ptbxl_metadata(cfg.data.ptbxl_dir)
split = split_by_domain(metadata, "site", cfg.data.train_sites, cfg.data.test_sites)
label_to_idx = {c: i for i, c in enumerate(DIAGNOSTIC_SUPERCLASSES)}
idx_to_label = {i: c for c, i in label_to_idx.items()}

train_ds = PTBXLDataset(split.train_df, cfg.data.ptbxl_dir, label_to_idx)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)

encoder = SignalEncoder(embed_dim=cfg.model.embed_dim, channel_widths=tuple(cfg.model.channel_widths))
model = SignalOnlyClassifier(encoder, num_classes=len(label_to_idx), embed_dim=cfg.model.embed_dim, hidden_dim=cfg.model.get("hidden_dim")).to(device)
model.load_state_dict(torch.load("experiments/results/baseline_signal_only_site1v2/best_model.pt", map_location=device))
model.eval()
print("Loaded trained baseline.")"""
    ),
    md("## Extract embeddings and compute per-class entropy"),
    code(
"""embeddings, labels = extract_embeddings(model, train_loader, device)
print(f"Embeddings: {embeddings.shape}, labels: {labels.shape}")

result = identify_rare_classes(embeddings, labels, k=cfg.rare_class.k)

print("\\nPer-class entropy (theta):")
for cls_idx, theta in sorted(result.class_entropy.items()):
    flag = " <-- RARE" if cls_idx in result.rare_classes else ""
    print(f"  {idx_to_label[cls_idx]:6s} theta={theta:.4f}{flag}")

print(f"\\nMean entropy: {result.mean_entropy:.4f}, Std: {result.std_entropy:.4f}")
print(f"Rare classes (Definition 2 threshold = mean +/- 1 std): "
      f"{[idx_to_label[c] for c in result.rare_classes]}")"""
    ),
    md("## Visualize"),
    code(
"""labels_sorted = sorted(result.class_entropy.keys())
thetas = [result.class_entropy[c] for c in labels_sorted]
colors = ["crimson" if c in result.rare_classes else "steelblue" for c in labels_sorted]

plt.figure(figsize=(7, 4))
plt.bar([idx_to_label[c] for c in labels_sorted], thetas, color=colors)
plt.axhline(result.mean_entropy, color="gray", linestyle="--", label="mean")
plt.axhline(result.mean_entropy + result.std_entropy, color="gray", linestyle=":", label="mean ± 1 std")
plt.axhline(result.mean_entropy - result.std_entropy, color="gray", linestyle=":")
plt.ylabel("Class entropy (theta)")
plt.title("Per-class entropy — red bars are classes flagged as rare")
plt.legend()
plt.show()"""
    ),
    md(
"""## Compare against raw class frequency

Worth checking: is the "rare" class by entropy the same as the class with
fewest raw examples? If not, that's exactly the reference paper's point —
rarity isn't just about count, it's about how spread out/inconsistent the
class's representations are."""
    ),
    code(
"""counts = np.bincount(labels, minlength=len(label_to_idx))
for i, c in enumerate(counts):
    print(f"{idx_to_label[i]:6s} n={c:5d}  theta={result.class_entropy.get(i, float('nan')):.4f}")"""
    ),
    md(
"""## Next steps (not yet built)

This notebook establishes *which class is rare* and confirms the entropy
math (already unit-tested in `tests/test_entropy.py`) runs correctly on
real learned embeddings. Phases 3–6 (multimodal fusion, missing-modality
robustness, full ablation) are scaffolded in `src/models/fusion.py`,
`src/models/text_encoder.py`, and `src/training/modality_dropout.py` —
see each file's docstring for the implementation plan, and
`notebooks/04_multimodal_fusion.ipynb` onward for their placeholders."""
    ),
])


def scaffold_notebook(title, phase_desc, depends_on, blocked_on_files, plan_items):
    plan_md = "\n".join(f"{i+1}. {item}" for i, item in enumerate(plan_items))
    return make_notebook([
        md(
f"""# {title}

**Status: not yet implemented — scaffold only.**

{phase_desc}

**Depends on:** {depends_on}

**Blocked on these modules being implemented first:**
{chr(10).join('- `' + f + '`' for f in blocked_on_files)}

## Implementation plan

{plan_md}

Once the modules above are implemented, this notebook will follow the
same pattern as `02_baseline_signal_only.ipynb` /
`03_rare_class_identification.ipynb`: a setup cell, a training/analysis
call against a config in `configs/`, and result inspection cells reading
from `experiments/results/`."""
        ),
    ])


nb04 = scaffold_notebook(
    "04 — Phase 3: Multimodal Fusion (Scaffold)",
    "Will train the fusion model (signal + independent text) on one PTB-XL "
    "site and evaluate on another, directly comparable to notebook 02's "
    "baseline numbers — this is the notebook that produces the headline "
    "\"does genuine multimodal fusion help under domain shift\" result.",
    "`configs/fusion_full.yaml`, `notebooks/00_colab_setup.ipynb`",
    ["src/models/text_encoder.py", "src/models/fusion.py"],
    [
        "Implement `TextEncoder` (text_encoder.py) — start with a frozen pretrained encoder + linear projection.",
        "Implement `FusionClassifier` (fusion.py) with `fusion_strategy=\"concat\"`.",
        "Extend `src/training/trainer.py`'s `build_model`/`build_data` to construct the fusion model + text-aware dataset when a fusion config is passed (or add a small `src/training/trainer_fusion.py` if the loops diverge enough to warrant it).",
        "Run against `configs/fusion_full.yaml`, compare rare-class F1 directly to notebook 02's baseline result.",
    ],
)

nb05 = scaffold_notebook(
    "05 — Phase 5: Missing-Modality Robustness (Scaffold)",
    "Will train the fusion model with modality dropout, then evaluate it "
    "under three conditions on the target domain — both modalities present, "
    "signal only, text only — and report the degradation using "
    "`src.evaluation.metrics.modality_degradation`.",
    "Notebook 04 (Phase 3 fusion) must be implemented and working first.",
    ["src/training/modality_dropout.py", "src/models/fusion.py (missing-input support)"],
    [
        "Add missing-modality support to `FusionClassifier.forward`/`.embed` (accept `signal=None` / `input_ids=None`, substitute a learned placeholder embedding).",
        "Implement `ModalityDropout` (modality_dropout.py) and wire it into the training loop, gated by `cfg.modality_dropout.enabled`.",
        "Implement the three-condition eval loop from `configs/fusion_robust.yaml`'s `eval.conditions`.",
        "Run against `configs/fusion_robust.yaml`; compare against notebook 04's fusion-without-dropout-training numbers to see how much dropout training recovers under missing modalities.",
    ],
)

nb06 = scaffold_notebook(
    "06 — Phase 6: Full Ablation Suite (Scaffold)",
    "Will assemble the final results table comparing image/signal-only vs. "
    "text-only vs. full fusion vs. fusion-with-robustness-training, across "
    "every domain-shift pair, with significance testing — the table this "
    "whole project is building toward (proposal Section 6.2 step 6).",
    "Notebooks 02, 04, and 05 must all be implemented and have been run at least once.",
    ["scripts/run_ablation.py (not yet written)"],
    [
        "Write `scripts/run_ablation.py`: loops over every (model variant × domain-shift pair) combination, calling the appropriate training entry point for each, collecting all `metrics.json` files into one table.",
        "Add a paired significance test (e.g. bootstrap or paired t-test across domain-shift pairs) comparing fusion vs. baseline rare-class F1.",
        "Render the final ablation table (image-only / text-only / fused / fused-robust × rare-class F1, precision, recall, accuracy) and save it to `experiments/results/ablation_summary.csv`.",
        "Write up the results narrative referencing back to the proposal's expected outcomes (Section 7).",
    ],
)

notebooks = {
    "00_colab_setup.ipynb": nb00,
    "01_data_exploration_ptbxl.ipynb": nb01,
    "02_baseline_signal_only.ipynb": nb02,
    "03_rare_class_identification.ipynb": nb03,
    "04_multimodal_fusion.ipynb": nb04,
    "05_missing_modality_robustness.ipynb": nb05,
    "06_ablation_results.ipynb": nb06,
}

import os
os.makedirs(NB_DIR, exist_ok=True)
for filename, nb in notebooks.items():
    path = os.path.join(NB_DIR, filename)
    nbf.write(nb, path)
    print(f"Wrote {path}")
