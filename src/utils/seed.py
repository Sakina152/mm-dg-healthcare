"""One function. Call it once, at the top of every entry point.

Every notebook and script in this project calls `set_seed(cfg.seed)` before
touching data or building a model. This is the single place that knows how
to seed every RNG source we use — if we add a new source of randomness
later (e.g. a new library), it gets added here once, not hunted down across
six notebooks.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices).

    Also sets PyTorch to deterministic algorithm mode where possible.
    Full bit-for-bit determinism on GPU is not guaranteed by PyTorch even
    with this set (some CUDA ops are non-deterministic by design), but this
    gets us as close as practical while keeping training speed usable.

    Args:
        seed: the seed value. Default 42, matches every config file's
            default so results are comparable unless a config explicitly
            overrides it.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Best-effort determinism; warn instead of crash if an op has no
    # deterministic implementation, since some interpolation/pooling ops
    # used by later fusion encoders may not have one.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Pass to DataLoader(worker_init_fn=...) so each DataLoader worker
    process gets a distinct, reproducible seed derived from the base seed.
    Without this, multi-worker DataLoaders can silently reduce randomness
    diversity or, worse, correlate augmentation across workers.
    """
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
