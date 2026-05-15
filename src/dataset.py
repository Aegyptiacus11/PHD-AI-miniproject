"""Datasets and dataloaders for chest X-ray folders and tensor caches."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.config import cfg

ModelType = Literal["cnn", "mobilenet", "resnet18", "swintiny"]


def _transfer_transforms(augment: bool) -> transforms.Compose:
    """Shared 224x224 RGB-style pipeline for ImageNet transfer models."""
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    ops: list = [
        transforms.Resize((cfg.mobilenet_img_size, cfg.mobilenet_img_size)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if augment:
        ops.extend(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])
    return transforms.Compose(ops)


def get_transforms(model_type: ModelType, augment: bool) -> transforms.Compose:
    """
    Build train or test transforms for the given model type.

    Args:
        model_type: ``cnn`` (single channel) or transfer models (3-channel gray).
        augment: If True, apply random rotation and affine jitter.
    """
    if model_type == "cnn":
        ops: list = [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.Grayscale(num_output_channels=1),
        ]
        if augment:
            ops.extend(
                [
                    transforms.RandomRotation(8),
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                ]
            )
        ops.extend([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        return transforms.Compose(ops)

    if model_type in ("mobilenet", "resnet18", "swintiny"):
        return _transfer_transforms(augment)

    raise ValueError(f"Unknown model_type: {model_type}")


def _make_imagefolder(path: str | Path, transform: transforms.Compose) -> datasets.ImageFolder:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {p}")
    return datasets.ImageFolder(root=str(p), transform=transform)


class CachedSplitDataset(Dataset[tuple[torch.Tensor | PILImage, int]]):
    """Load split cache exported by ``src.export_cache`` and apply runtime transforms."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        transform: transforms.Compose | None = None,
        expected_classes: list[str] | None = None,
    ) -> None:
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        self._images: torch.Tensor = data["images"].to(torch.uint8).cpu()
        self._labels: torch.Tensor = data["labels"].long().cpu()
        self.classes: list[str] = [str(c) for c in data.get("classes", [])]
        self.transform = transform
        if self._images.ndim != 4 or self._images.size(1) != 1:
            raise ValueError(f"Cache images must be N x 1 x H x W uint8, got shape={tuple(self._images.shape)}")
        if self._labels.ndim != 1 or len(self._labels) != len(self._images):
            raise ValueError(
                f"Cache labels must be 1D with same length as images, got images={len(self._images)} labels={len(self._labels)}"
            )
        if expected_classes is not None and self.classes and self.classes != expected_classes:
            raise ValueError(f"Class mismatch in cache. expected={expected_classes} got={self.classes}")

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | PILImage, int]:
        pixels = self._images[idx].squeeze(0).numpy()
        image: torch.Tensor | PILImage = Image.fromarray(pixels, mode="L")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self._labels[idx].item())


def _resolve_eval_cache_path(
    *,
    val_dir: str | Path | None,
    cache_val_pt: Path,
    cache_test_pt: Path,
) -> Path:
    if val_dir is not None and Path(val_dir).is_dir():
        return cache_val_pt
    return cache_test_pt


def get_dataloaders(
    train_dir: str | Path,
    test_dir: str | Path,
    model_type: ModelType,
    *,
    val_dir: str | Path | None = None,
    use_cache: bool = False,
    cache_train_pt: str | Path | None = None,
    cache_val_pt: str | Path | None = None,
    cache_test_pt: str | Path | None = None,
) -> tuple[DataLoader[tuple[torch.Tensor, int]], DataLoader[tuple[torch.Tensor, int]]]:
    """
    Create train and validation/test DataLoaders with augmentation on train only.

    If ``val_dir`` is provided, it is used as the validation loader; otherwise test split is reused.
    Cache loading is opt-in via ``use_cache``.
    """
    train_tf = get_transforms(model_type, augment=True)
    eval_tf = get_transforms(model_type, augment=False)

    if use_cache:
        c_train = Path(cache_train_pt or cfg.cache_train_pt)
        c_val = Path(cache_val_pt or cfg.cache_val_pt)
        c_test = Path(cache_test_pt or cfg.cache_test_pt)
        c_eval = _resolve_eval_cache_path(val_dir=val_dir, cache_val_pt=c_val, cache_test_pt=c_test)
        missing = [p for p in (c_train, c_eval) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cache mode requested (--use_cache) but required cache file(s) are missing: "
                + ", ".join(str(p) for p in missing)
            )
        train_ds = CachedSplitDataset(c_train, transform=train_tf)
        test_ds: Dataset[tuple[torch.Tensor | PILImage, int]] = CachedSplitDataset(
            c_eval, transform=eval_tf, expected_classes=train_ds.classes or None
        )
    else:
        train_ds = _make_imagefolder(train_dir, train_tf)
        eval_root = val_dir if val_dir is not None else test_dir
        test_ds = _make_imagefolder(eval_root, eval_tf)

    train_classes = train_ds.classes if hasattr(train_ds, "classes") else []
    eval_classes = test_ds.classes if hasattr(test_ds, "classes") else []
    if train_classes and eval_classes and train_classes != eval_classes:
        raise ValueError(f"Class mismatch between splits. train={train_classes} eval={eval_classes}")
    if train_classes and len(train_classes) != cfg.num_classes:
        raise ValueError(
            f"Expected {cfg.num_classes} classes but found {len(train_classes)} in {train_dir}: {train_classes}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader
