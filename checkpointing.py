"""Resolve which .pt file to load for evaluation."""

from pathlib import Path

from train import best_checkpoint_path, checkpoint_path


def resolve_checkpoint(
    cfg,
    hybrid: bool,
    strategy: str = "auto",
    num_epochs: int = 10,
) -> Path:
    """
    strategy:
      auto — best validation ckpt if present, else epoch num_epochs, else latest epoch
      best — best only, then fall back like auto
      last — epoch num_epochs, then latest (ignore best)
    """
    best = best_checkpoint_path(hybrid, cfg)
    epoch_ckpt = checkpoint_path(num_epochs, hybrid, cfg)
    pattern = "coast_hybrid_epoch*.pt" if hybrid else "coast_epoch*.pt"
    all_ckpts = sorted(cfg.checkpoint_dir().glob(pattern))

    if strategy in ("auto", "best") and best.is_file():
        print("using best checkpoint", best)
        return best

    if strategy == "last" or strategy == "auto":
        if epoch_ckpt.is_file():
            print("using checkpoint", epoch_ckpt)
            return epoch_ckpt
        if all_ckpts:
            ckpt = all_ckpts[-1]
            print("using latest checkpoint", ckpt)
            return ckpt

    if strategy == "best" and epoch_ckpt.is_file():
        print("best not found; using checkpoint", epoch_ckpt)
        return epoch_ckpt

    raise FileNotFoundError(
        f"no checkpoint for dataset={cfg.name!r} under {cfg.checkpoint_dir()}. "
        f"Run: python main.py --dataset {cfg.name} --mode train --device cuda"
    )
