from coast.train.loop import best_checkpoint_path, checkpoint_path


def resolve_checkpoint(cfg, hybrid: bool, strategy: str = "auto", num_epochs: int = 10):
    best = best_checkpoint_path(hybrid, cfg)
    if strategy in ("auto", "best") and best.is_file():
        print("using best checkpoint", best)
        return best

    pattern = "coast_hybrid_epoch*.pt" if hybrid else "coast_epoch*.pt"
    epoch_ckpts = sorted(cfg.checkpoint_dir().glob(pattern))
    if epoch_ckpts:
        print("using checkpoint", epoch_ckpts[-1])
        return epoch_ckpts[-1]

    raise FileNotFoundError(
        f"no checkpoint for dataset={cfg.name!r}. "
        f"Run: python -m coast.cli.main --dataset {cfg.name} --mode train --device cuda"
    )
