"""Device selection and reproducibility controls for local ML experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def select_device(requested: str = "auto") -> torch.device:
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if requested == "auto":
        for name in ("cuda", "mps", "cpu"):
            if available[name]:
                return torch.device(name)
    if requested not in available:
        raise ValueError(f"Unsupported device request {requested!r}.")
    if not available[requested]:
        raise RuntimeError(f"Requested device {requested!r} is not available.")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. MPS kernels may still vary by runtime."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU execution can use deterministic algorithms. Some accelerators do not
    # implement every deterministic kernel, so this is a best-effort control.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
