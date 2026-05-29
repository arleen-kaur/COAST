"""Runtime version checks so Colab/local stay compatible."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_version() -> str:
    vfile = ROOT / "VERSION"
    if not vfile.is_file():
        return "unknown"
    return vfile.read_text().strip()


COAST_VERSION = read_version()


def check_installation() -> None:
    """Fail fast with a clear message if the repo is missing Phase 2 features."""
    import train

    missing = []
    if not hasattr(train, "best_checkpoint_path"):
        missing.append("train.best_checkpoint_path (early stopping)")
    if not hasattr(train, "train_loop") or "early_stop" not in train.train_loop.__code__.co_varnames:
        pass  # optional; train_loop may not expose early_stop in co_varnames reliably

    if not (ROOT / "checkpointing.py").is_file():
        missing.append("checkpointing.py (auto checkpoint resolution)")

    if missing:
        raise SystemExit(
            f"\nCOAST install is outdated (local VERSION={COAST_VERSION}).\n"
            "Missing: " + ", ".join(missing) + "\n\n"
            "On Colab or any machine, sync with GitHub:\n"
            "  %cd /content/COAST   # or your clone path\n"
            "  !git pull origin main\n"
            "  !python scripts/verify_setup.py\n"
        )


def print_banner() -> None:
    print(f"COAST v{COAST_VERSION} | https://github.com/arleen-kaur/COAST")
