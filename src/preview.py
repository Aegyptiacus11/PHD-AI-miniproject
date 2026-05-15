"""CLI to preview chest X-ray dataset folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from src.config import cfg


def _imagefolder(split_dir: str | Path) -> datasets.ImageFolder:
    p = Path(split_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Split directory not found: {p}")
    return datasets.ImageFolder(root=str(p), transform=transforms.Grayscale(num_output_channels=1))


def _print_stats(
    train_dir: str | Path,
    test_dir: str | Path | None,
    val_dir: str | Path | None,
) -> None:
    """Print dataset sizes for available splits."""
    n_train = len(_imagefolder(train_dir))
    print(f"train_samples={n_train}", flush=True)
    if val_dir is not None:
        n_val = len(_imagefolder(val_dir))
        print(f"val_samples={n_val}", flush=True)
    if test_dir is not None:
        n_test = len(_imagefolder(test_dir))
        print(f"test_samples={n_test}", flush=True)


def _save_sample_grid(
    train_dir: str | Path,
    out_path: Path,
    n: int,
    dpi: int,
) -> None:
    """Save a square grid of the first ``n`` training samples."""
    ds = _imagefolder(train_dir)
    n_show = min(n, len(ds))
    side = int(np.ceil(np.sqrt(n_show)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(side, side, figsize=(2.2 * side, 2.2 * side), dpi=dpi)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat:
        ax.axis("off")
    for i in range(n_show):
        img, lab = ds[i]
        ax = axes_flat[i]
        ax.imshow(np.asarray(img).squeeze(), cmap="gray")
        label_name = ds.classes[lab] if 0 <= int(lab) < len(ds.classes) else str(lab)
        ax.set_title(label_name, fontsize=9)
    fig.suptitle("Sample training images (chest_xray/train)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved preview grid to {out_path}", flush=True)


def _save_problem_statement_figures(train_dir: str | Path, out_dir: str | Path, *, dpi: int) -> None:
    """Export one NORMAL and one PNEUMONIA training example for report introductions."""
    ds = _imagefolder(train_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = {"NORMAL": None, "PNEUMONIA": None}
    for name in targets:
        if name not in ds.class_to_idx:
            raise ValueError(f"Expected class folder {name!r} under {train_dir}, got {list(ds.class_to_idx)}")
    want = {ds.class_to_idx["NORMAL"]: "NORMAL", ds.class_to_idx["PNEUMONIA"]: "PNEUMONIA"}
    for i in range(len(ds)):
        _, lab = ds[i]
        key = want.get(int(lab))
        if key is not None and targets[key] is None:
            targets[key] = i
        if all(v is not None for v in targets.values()):
            break
    missing = [k for k, v in targets.items() if v is None]
    if missing:
        raise RuntimeError(f"Could not find training samples for classes: {missing}")

    for label, idx in targets.items():
        assert idx is not None
        img, _ = ds[idx]
        arr = np.asarray(img).squeeze()
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=dpi)
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
        fig.tight_layout(pad=0.1)
        path = out / f"problem_statement_{label.lower()}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved {path}", flush=True)


def run_preview(
    train_dir: str | Path,
    *,
    test_dir: str | Path | None = None,
    val_dir: str | Path | None = None,
    out: str | Path | None = None,
    n: int = 16,
    dpi: int = 150,
    stats: bool = False,
) -> None:
    """Programmatic entry point (e.g. notebooks): same behavior as the ``src.preview`` CLI (without argparse)."""
    if stats:
        _print_stats(train_dir, test_dir, val_dir)
        return
    _print_stats(train_dir, test_dir, val_dir)
    out_path = Path(out or Path(cfg.results_dir) / "dataset_preview.png")
    _save_sample_grid(train_dir, out_path, n=n, dpi=dpi)


def run_problem_statement_export(train_dir: str | Path, problem_statement_dir: str | Path, *, dpi: int) -> None:
    """Export introductory CXR figures (one per class) for the LaTeX problem statement."""
    _save_problem_statement_figures(train_dir, problem_statement_dir, dpi=dpi)


def main() -> None:
    """Parse CLI arguments and write the preview figure (and optional stats-only mode)."""
    parser = argparse.ArgumentParser(description="Preview chest X-ray dataset folders.")
    parser.add_argument("--train_dir", type=str, required=True, help="Training directory.")
    parser.add_argument("--test_dir", type=str, default=None, help="Optional test directory for counts.")
    parser.add_argument("--val_dir", type=str, default=None, help="Optional validation directory for counts.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(cfg.results_dir) / "dataset_preview.png"),
        help="Output image path for the sample grid.",
    )
    parser.add_argument("--n", type=int, default=16, help="Number of samples in the grid.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Only print train/test row counts; do not write an image.",
    )
    parser.add_argument(
        "--problem_statement_dir",
        type=str,
        default=None,
        help="If set, also write problem_statement_normal.png and problem_statement_pneumonia.png there.",
    )
    args = parser.parse_args()

    if args.problem_statement_dir:
        _save_problem_statement_figures(args.train_dir, args.problem_statement_dir, dpi=args.dpi)
    run_preview(
        args.train_dir,
        test_dir=args.test_dir,
        val_dir=args.val_dir,
        out=args.out,
        n=args.n,
        dpi=args.dpi,
        stats=args.stats,
    )


if __name__ == "__main__":
    main()
