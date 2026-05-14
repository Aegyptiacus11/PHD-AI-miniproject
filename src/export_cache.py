"""Export chest X-ray image-folder splits to tensor caches (.pt)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from src.config import cfg


def _to_uint8_gray(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as im:
        gray = im.convert("L").resize((image_size, image_size), resample=Image.BILINEAR)
        return np.asarray(gray, dtype=np.uint8)


def export_split(split_dir: str | Path, out_path: str | Path, *, image_size: int) -> None:
    """Write one split as uint8 image tensors and labels."""
    split_dir = Path(split_dir)
    out_path = Path(out_path)
    ds = datasets.ImageFolder(root=str(split_dir), transform=transforms.Grayscale(num_output_channels=1))
    images: list[np.ndarray] = []
    labels: list[int] = []
    rel_paths: list[str] = []
    for sample_path, target in ds.samples:
        images.append(_to_uint8_gray(Path(sample_path), image_size=image_size))
        labels.append(int(target))
        rel_paths.append(str(Path(sample_path).relative_to(split_dir)))

    if not images:
        raise ValueError(f"No images found in split directory: {split_dir}")

    image_np = np.stack(images, axis=0)  # N x H x W
    image_t = torch.from_numpy(image_np.copy()).unsqueeze(1).contiguous()  # N x 1 x H x W
    label_t = torch.tensor(labels, dtype=torch.long)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "images": image_t,
            "labels": label_t,
            "classes": list(ds.classes),
            "split_dir": str(split_dir),
            "image_size": int(image_size),
            "paths": rel_paths,
        },
        out_path,
    )
    print(f"Wrote {out_path} (N={len(ds)} shape={tuple(image_t.shape)})", flush=True)


def main() -> None:
    """CLI: export train/val/test split caches as .pt files."""
    parser = argparse.ArgumentParser(description="Export chest X-ray image folders to tensor cache (.pt).")
    parser.add_argument("--train_dir", type=str, default=cfg.train_dir)
    parser.add_argument("--test_dir", type=str, default=cfg.test_dir)
    parser.add_argument("--val_dir", type=str, default=cfg.val_dir)
    parser.add_argument("--out_train", type=str, default=cfg.cache_train_pt)
    parser.add_argument("--out_test", type=str, default=cfg.cache_test_pt)
    parser.add_argument("--out_val", type=str, default=cfg.cache_val_pt)
    parser.add_argument("--image_size", type=int, default=cfg.mobilenet_img_size)
    args = parser.parse_args()
    export_split(args.train_dir, args.out_train, image_size=args.image_size)
    export_split(args.test_dir, args.out_test, image_size=args.image_size)
    val_dir = Path(args.val_dir)
    if val_dir.is_dir():
        export_split(val_dir, args.out_val, image_size=args.image_size)
    else:
        print(f"Skipping val export (directory not found): {val_dir}", flush=True)


if __name__ == "__main__":
    main()
