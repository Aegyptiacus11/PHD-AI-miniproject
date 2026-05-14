"""Resolve chest X-ray split directories on Kaggle and local."""

from __future__ import annotations

import os
from pathlib import Path

_SPLITS = ("train", "test")
_CLASS_DIRS = ("NORMAL", "PNEUMONIA")


def resolve_repo_root() -> Path:
    """Project root: cwd if it looks like the repo, else common Kaggle clone paths."""
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "src").is_dir():
        return cwd
    for p in (
        Path("/kaggle/working/PHD-AI-miniproject"),
        Path("/kaggle/working/sign-language-classifier"),
    ):
        if (p / "pyproject.toml").is_file() and (p / "src").is_dir():
            return p
    return cwd


def _is_valid_chest_xray_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    for split in _SPLITS:
        split_dir = path / split
        if not split_dir.is_dir():
            return False
        if not all((split_dir / cls).is_dir() for cls in _CLASS_DIRS):
            return False
    return True


def find_kaggle_chest_xray_dir() -> Path | None:
    """Return a directory under /kaggle/input that contains train/test split folders."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None

    env = os.environ.get("CHEST_XRAY_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        if _is_valid_chest_xray_root(p):
            return p

    for p in base.rglob("chest_xray"):
        if _is_valid_chest_xray_root(p):
            return p
    return None


def resolve_chest_xray_dirs() -> tuple[str, str, str | None]:
    """Return (train_dir, test_dir, val_dir)."""
    d = find_kaggle_chest_xray_dir()
    if d is not None:
        val_dir = d / "val"
        return str(d / "train"), str(d / "test"), (str(val_dir) if val_dir.is_dir() else None)
    if Path("/kaggle").is_dir():
        raise FileNotFoundError(
            "On Kaggle but could not find a `chest_xray/` directory with `train/` and `test/` "
            "splits under /kaggle/input. Add the dataset "
            "`paultimothymooney/chest-xray-pneumonia` or set CHEST_XRAY_DIR to that folder."
        )
    return "data/chest_xray/train", "data/chest_xray/test", "data/chest_xray/val"
