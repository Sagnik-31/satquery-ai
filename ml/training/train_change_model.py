"""Train the v1 Siamese U-Net using official LEVIR-CD train/val only.

This command is intentionally separate from the FastAPI backend and has no
argument or fallback capable of resolving the locked test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.datasets.levir_cd import LevirPairedDataset, compute_train_rgb_normalization
from ml.datasets.paired_transforms import PairedSpatialTransform
from ml.evaluation.evaluate_model import evaluate_validation_model
from ml.models.siamese_unet import SiameseUNet
from ml.training.checkpointing import save_checkpoint
from ml.training.config import TrainingConfig, load_training_config
from ml.training.reproducibility import seed_everything, select_device


class SoftDiceLoss(nn.Module):
    """Soft binary Dice loss over a batch of logits and binary masks."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        target = target.flatten(1)
        intersection = (probabilities * target).sum(dim=1)
        denominator = probabilities.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 1.0 - dice.mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The fixed v1 objective: 0.5 BCEWithLogits + 0.5 Soft Dice."""
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * SoftDiceLoss()(logits, target)


def _git_revision(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_metadata(
    run_dir: Path,
    *,
    config: TrainingConfig,
    normalization: dict[str, list[float]],
    train_dataset: LevirPairedDataset,
    val_dataset: LevirPairedDataset,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n")
    (run_dir / "normalization.json").write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n")
    manifest = {
        "train": [sample.identifier for sample in train_dataset.samples],
        "val": [sample.identifier for sample in val_dataset.samples],
        "git_revision": _git_revision(Path(__file__).resolve().parents[2]),
        "test_split_accessed": False,
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _checkpoint_provenance(config: TrainingConfig) -> dict[str, Any]:
    config_json = json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))
    return {
        "architecture": "SiameseUNet",
        "channel_schema": {"t1": config.model.input_channels, "t2": config.model.input_channels, "output_logits": 1},
        "git_revision": _git_revision(Path(__file__).resolve().parents[2]),
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "selection_metric": "global_validation_iou",
    }


def train(
    *,
    dataset_root: str | Path,
    config: TrainingConfig,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Run train/validation training. The caller must explicitly start this."""
    seed_everything(config.seed)
    device = select_device(config.device)
    normalization = compute_train_rgb_normalization(dataset_root)
    paired_transform = PairedSpatialTransform(patch_size=config.patch_size)
    train_dataset = LevirPairedDataset(
        dataset_root,
        split="train",
        transform=paired_transform,
        normalization=normalization,
        seed=config.seed,
    )
    val_dataset = LevirPairedDataset(dataset_root, split="val", normalization=normalization)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    model = SiameseUNet(
        input_channels=config.model.input_channels,
        base_channels=config.model.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler_t_max)
    destination = Path(run_dir)
    normalization_dict = normalization.as_dict()
    provenance = _checkpoint_provenance(config)
    _write_metadata(destination, config=config, normalization=normalization_dict, train_dataset=train_dataset, val_dataset=val_dataset)

    best_iou = float("-inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            target = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(t1, t2), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        validation = evaluate_validation_model(
            model,
            val_dataset,
            device=device,
            patch_size=config.patch_size,
            stride=config.patch_size // 2,
            threshold=config.threshold,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
        }
        history.append(record)
        save_checkpoint(
            destination / "checkpoints" / f"epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config.as_dict(),
            normalization=normalization_dict,
            validation=validation,
            provenance=provenance,
        )
        if validation["iou"] > best_iou:
            best_iou = float(validation["iou"])
            save_checkpoint(
                destination / "checkpoints" / "best_val_iou.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config.as_dict(),
                normalization=normalization_dict,
                validation=validation,
                provenance=provenance,
            )
        (destination / "train_log.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"device": str(device), "best_validation_iou": best_iou, "epochs": config.epochs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Local LEVIR-CD root; only train/ and val/ are read.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, help="Ignored local output directory for one experiment run.")
    args = parser.parse_args()
    result = train(dataset_root=args.dataset_root, config=load_training_config(args.config), run_dir=args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
