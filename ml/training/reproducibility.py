"""Device selection and reproducibility controls for local ML experiments."""

from __future__ import annotations

import random
import platform
import sys
from importlib import metadata

import numpy as np
import torch


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _package_snapshot() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


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


def runtime_metadata(device: torch.device) -> dict[str, object]:
    """Portable runtime record; MPS reproducibility remains best-effort."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "macos_version": platform.mac_ver()[0] or None,
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "torch_version": torch.__version__,
        "torchvision_version": _package_version("torchvision"),
        "opencv_version": _package_version("opencv-python-headless") or _package_version("opencv-python"),
        "rasterio_version": _package_version("rasterio"),
        "selected_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "mps_reproducibility": "best_effort_not_bitwise_guaranteed",
        "package_snapshot": _package_snapshot(),
    }
