"""Generic train/eval loop, entry point for Phase 1.

Usage:
    python -m src.training.trainer --config configs/baseline_signal_only.yaml

Deliberately model-agnostic: it takes any `nn.Module` whose `forward`
returns logits and any `Dataset` whose `__getitem__` returns a dict with
"signal" (or later, "image"/"text") and "label" keys, so Phase 3's fusion
model plugs into the exact same loop without a rewrite — only the model
and dataset constructed in `build_model` / `build_data` change.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.domain_split import split_by_domain
from src.data.ptbxl_dataset import DIAGNOSTIC_SUPERCLASSES, PTBXLDataset, load_ptbxl_metadata
from src.evaluation.metrics import compute_metrics
from src.models.classifier import SignalOnlyClassifier
from src.models.signal_encoder import SignalEncoder
from src.rare_class.entropy import identify_rare_classes
from src.utils.config import Config, load_config, save_config
from src.utils.seed import set_seed, worker_init_fn


def build_data(cfg: Config):
    """Load PTB-XL metadata, apply the domain-shift split, and wrap both
    sides in `PTBXLDataset` + `DataLoader`.
    """
    metadata = load_ptbxl_metadata(cfg.data.ptbxl_dir)
    split = split_by_domain(
        metadata,
        domain_col="site",
        train_domains=cfg.data.train_sites,
        test_domains=cfg.data.test_sites,
    )
    print(split.summary())

    label_to_idx = {c: i for i, c in enumerate(DIAGNOSTIC_SUPERCLASSES)}
    train_ds = PTBXLDataset(split.train_df, cfg.data.ptbxl_dir, label_to_idx)
    test_ds = PTBXLDataset(split.test_df, cfg.data.ptbxl_dir, label_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.get("num_workers", 2),
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.get("num_workers", 2),
    )
    return train_loader, test_loader, label_to_idx


def build_model(cfg: Config, num_classes: int) -> torch.nn.Module:
    encoder = SignalEncoder(
        in_channels=12,
        embed_dim=cfg.model.embed_dim,
        channel_widths=tuple(cfg.model.channel_widths),
    )
    return SignalOnlyClassifier(
        encoder=encoder,
        num_classes=num_classes,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.get("hidden_dim"),
    )


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            signal = batch["signal"].to(device)
            label = batch["label"].to(device)

            logits = model(signal)
            loss = criterion(logits, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * signal.size(0)
            all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return avg_loss, preds, labels


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Pull encoder embeddings + labels for the whole loader — used by
    Phase 2 (rare-class identification) so it can run on the trained
    baseline's actual learned representation, not raw signal statistics.
    """
    model.eval()
    all_emb, all_labels = [], []
    for batch in loader:
        signal = batch["signal"].to(device)
        emb = model.embed(signal)
        all_emb.append(emb.cpu().numpy())
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_emb), np.concatenate(all_labels)


def train(cfg: Config) -> dict:
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, label_to_idx = build_data(cfg)
    num_classes = len(label_to_idx)
    model = build_model(cfg, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.get("weight_decay", 0.0))
    criterion = torch.nn.CrossEntropyLoss()

    run_dir = Path(cfg.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config_used.yaml")  # snapshot: results never separated from settings

    history = []
    best_rare_f1 = -1.0
    for epoch in range(cfg.train.epochs):
        t0 = time.time()
        train_loss, train_preds, train_labels = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        test_loss, test_preds, test_labels = run_epoch(model, test_loader, optimizer, criterion, device, train=False)

        # Rare class is identified once per run, on this run's TRAIN-domain
        # embeddings (matching Definition 2's intent: rarity is a property
        # of the source domain the model learns from), then reused across
        # epochs so the "rare class" definition doesn't drift mid-training.
        if epoch == cfg.train.epochs - 1:
            train_emb, train_emb_labels = extract_embeddings(model, train_loader, device)
            entropy_result = identify_rare_classes(train_emb, train_emb_labels, k=cfg.rare_class.get("k", 5))
            rare_labels = entropy_result.rare_classes or [int(np.bincount(train_labels).argmin())]
        else:
            rare_labels = [int(np.bincount(train_labels).argmin())]  # cheap proxy during training

        test_metrics = compute_metrics(
            test_labels, test_preds, class_labels=list(range(num_classes)), rare_class_labels=rare_labels
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_metrics": test_metrics.to_dict(),
            "elapsed_sec": time.time() - t0,
        }
        history.append(epoch_record)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} test_loss={test_loss:.4f} {test_metrics}")

        if test_metrics.rare_class_f1 > best_rare_f1:
            best_rare_f1 = test_metrics.rare_class_f1
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    final = history[-1]["test_metrics"]
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nDone. Results written to {run_dir}/")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
