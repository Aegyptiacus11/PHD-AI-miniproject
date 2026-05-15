"""Training CLI using PyTorch Lightning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.strategies import DDPStrategy

from src.config import cfg, checkpoint_path_for_model
from src.dataset import get_dataloaders
from src.lit_module import BestValCheckpointCallback, LitSignClassifier
from src.model import ModelName


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) for reproducibility."""
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _merge_histories(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for k in a:
        out[k] = list(a[k]) + list(b.get(k, []))
    return out


def _trainer_kwargs() -> dict[str, Any]:
    """Build ``Trainer`` kwargs: multi-GPU DDP when several CUDA devices are visible (e.g. Kaggle 2×T4)."""
    n_cuda = torch.cuda.device_count()
    base = {
        "logger": False,
        "enable_checkpointing": False,
        "deterministic": True,
    }
    if not torch.cuda.is_available() or n_cuda == 0:
        return {**base, "accelerator": "cpu", "devices": 1}

    want = cfg.trainer_num_devices
    num_dev = n_cuda if want < 1 else min(int(want), n_cuda)
    out: dict[str, Any] = {**base, "accelerator": "gpu", "devices": num_dev}
    if num_dev > 1:
        # Frozen backbone phases omit gradients on backbone params; DDP needs this flag.
        out["strategy"] = DDPStrategy(find_unused_parameters=True)
    return out


def _make_trainer(*, max_epochs: int, callbacks: list[Any]) -> L.Trainer:
    """
    Build a fresh ``Trainer`` instance.

    A new DDPStrategy instance is required for each Trainer; reusing one across
    phases triggers Lightning misconfiguration errors.
    """
    return L.Trainer(max_epochs=max_epochs, callbacks=callbacks, **_trainer_kwargs())


def fit_model_lightning(
    model_type: ModelName,
    train_dir: str | Path,
    test_dir: str | Path,
    *,
    val_dir: str | Path | None = None,
    use_cache: bool = False,
    cache_train_pt: str | Path | None = None,
    cache_val_pt: str | Path | None = None,
    cache_test_pt: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[dict[str, Any], float, int]:
    """
    Train one model with Lightning (two-phase for transfer nets).

    Returns:
        ``(history_entry, best_val_acc, best_epoch_index)``
    """
    set_seed(cfg.seed)
    train_loader, val_loader = get_dataloaders(
        train_dir,
        test_dir,
        model_type,
        val_dir=val_dir,
        use_cache=use_cache,
        cache_train_pt=cache_train_pt,
        cache_val_pt=cache_val_pt,
        cache_test_pt=cache_test_pt,
    )
    ckpt_path = str(checkpoint_path) if checkpoint_path is not None else checkpoint_path_for_model(model_type)
    if model_type == "cnn":
        lit = LitSignClassifier(
            model_type,
            freeze_backbone=False,
            max_epochs_this_phase=cfg.num_epochs,
        )
        cb = BestValCheckpointCallback(ckpt_path, model_type, epoch_offset=0)
        trainer = _make_trainer(max_epochs=cfg.num_epochs, callbacks=[cb])
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
        hist = {**lit.history}
        return _pack_history(hist, cb.best_val, int(cb.best_epoch))

    e1 = cfg.freeze_backbone_epochs
    e2 = max(0, cfg.num_epochs - e1)

    if e1 <= 0:
        lit = LitSignClassifier(model_type, freeze_backbone=False, max_epochs_this_phase=cfg.num_epochs)
        cb = BestValCheckpointCallback(ckpt_path, model_type, epoch_offset=0)
        trainer = _make_trainer(max_epochs=cfg.num_epochs, callbacks=[cb])
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return _pack_history({**lit.history}, cb.best_val, int(cb.best_epoch))

    shared_cb = BestValCheckpointCallback(ckpt_path, model_type, epoch_offset=0)

    lit1 = LitSignClassifier(model_type, freeze_backbone=True, max_epochs_this_phase=e1)
    t1 = _make_trainer(max_epochs=e1, callbacks=[shared_cb])
    t1.fit(lit1, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if e2 == 0:
        hist = {**lit1.history}
        return _pack_history(hist, shared_cb.best_val, int(shared_cb.best_epoch))

    lit2 = LitSignClassifier(model_type, freeze_backbone=False, max_epochs_this_phase=e2)
    lit2.net.load_state_dict(lit1.net.state_dict())
    shared_cb.epoch_offset = e1
    t2 = _make_trainer(max_epochs=e2, callbacks=[shared_cb])
    t2.fit(lit2, train_dataloaders=train_loader, val_dataloaders=val_loader)

    hist = _merge_histories(lit1.history, lit2.history)
    return _pack_history(hist, shared_cb.best_val, int(shared_cb.best_epoch))


def _pack_history(hist: dict[str, list[float]], best_val: float, best_epoch: int) -> tuple[dict[str, Any], float, int]:
    entry: dict[str, Any] = {**hist, "best_val_acc": float(best_val), "best_epoch": int(best_epoch)}
    return entry, float(best_val), int(best_epoch)


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.lr is not None:
        cfg.learning_rate = float(args.lr)
    if args.mobilenet_lr is not None:
        cfg.mobilenet_lr = float(args.mobilenet_lr)
    if args.resnet_lr is not None:
        cfg.resnet_lr = float(args.resnet_lr)
    if args.swintiny_lr is not None:
        cfg.swintiny_lr = float(args.swintiny_lr)
    if args.num_epochs is not None:
        cfg.num_epochs = int(args.num_epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.weight_decay is not None:
        cfg.weight_decay = float(args.weight_decay)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.freeze_backbone_epochs is not None:
        cfg.freeze_backbone_epochs = int(args.freeze_backbone_epochs)
    if args.devices is not None:
        cfg.trainer_num_devices = int(args.devices)
    if args.cache_train_pt is not None:
        cfg.cache_train_pt = args.cache_train_pt
    if args.cache_val_pt is not None:
        cfg.cache_val_pt = args.cache_val_pt
    if args.cache_test_pt is not None:
        cfg.cache_test_pt = args.cache_test_pt


def run_training(
    *,
    model: str = "all",
    train_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    val_dir: str | Path | None = None,
    lr: float | None = None,
    mobilenet_lr: float | None = None,
    resnet_lr: float | None = None,
    swintiny_lr: float | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    weight_decay: float | None = None,
    seed: int | None = None,
    freeze_backbone_epochs: int | None = None,
    use_cache: bool = False,
    cache_train_pt: str | None = None,
    cache_val_pt: str | None = None,
    cache_test_pt: str | None = None,
    devices: int | None = None,
    run_name: str | None = None,
) -> None:
    """
    Programmatic entry point (e.g. notebooks): same behavior as the ``src.train`` CLI without subprocesses.
    """
    if lr is not None:
        cfg.learning_rate = float(lr)
    if mobilenet_lr is not None:
        cfg.mobilenet_lr = float(mobilenet_lr)
    if resnet_lr is not None:
        cfg.resnet_lr = float(resnet_lr)
    if swintiny_lr is not None:
        cfg.swintiny_lr = float(swintiny_lr)
    if num_epochs is not None:
        cfg.num_epochs = int(num_epochs)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if weight_decay is not None:
        cfg.weight_decay = float(weight_decay)
    if seed is not None:
        cfg.seed = int(seed)
    if freeze_backbone_epochs is not None:
        cfg.freeze_backbone_epochs = int(freeze_backbone_epochs)
    if devices is not None:
        cfg.trainer_num_devices = int(devices)
    if cache_train_pt is not None:
        cfg.cache_train_pt = cache_train_pt
    if cache_val_pt is not None:
        cfg.cache_val_pt = cache_val_pt
    if cache_test_pt is not None:
        cfg.cache_test_pt = cache_test_pt

    tr = Path(train_dir or cfg.train_dir)
    te = Path(test_dir or cfg.test_dir)
    va = Path(val_dir) if val_dir is not None else (Path(cfg.val_dir) if Path(cfg.val_dir).is_dir() else None)

    tk = _trainer_kwargs()
    print(
        f"Lightning accelerator={tk.get('accelerator')} devices={tk.get('devices')} "
        f"strategy={type(tk.get('strategy')).__name__ if tk.get('strategy') is not None else 'default'}",
        flush=True,
    )

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    models_to_run: list[ModelName] = (
        list(cfg.compare_models) if model == "all" else [model]  # type: ignore[list-item]
    )

    all_histories: dict[str, dict[str, Any]] = {}
    for name in models_to_run:
        run_key = run_name if run_name and len(models_to_run) == 1 else name
        if run_name and len(models_to_run) > 1:
            run_key = f"{run_name}_{name}"
        ckpt_out = (
            str(Path(cfg.model_dir) / f"{run_key}_best.pth")
            if run_name
            else checkpoint_path_for_model(name)
        )
        print(f"\n=== Training {name} ===", flush=True)
        entry, best_val, best_ep = fit_model_lightning(
            name,
            tr,
            te,
            val_dir=va,
            use_cache=use_cache,
            cache_train_pt=cfg.cache_train_pt,
            cache_val_pt=cfg.cache_val_pt,
            cache_test_pt=cfg.cache_test_pt,
            checkpoint_path=ckpt_out,
        )
        hist_only = {
            k: v
            for k, v in entry.items()
            if k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")
        }
        row: dict[str, Any] = {
            **hist_only,
            "best_val_acc": best_val,
            "best_epoch": best_ep,
            "model_type": name,
            "checkpoint_path": ckpt_out,
        }
        all_histories[run_key] = row
        print(
            f"[{run_key}] finished best_val_acc={best_val:.4f} best_epoch={best_ep + 1} ckpt={ckpt_out}",
            flush=True,
        )

    out_path = Path(cfg.histories_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_histories: dict[str, dict[str, Any]] = {}
    if out_path.is_file():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                merged_histories = {k: v for k, v in prev.items() if isinstance(v, dict)}
        except json.JSONDecodeError:
            print(f"Warning: could not parse existing histories at {out_path}; overwriting.", flush=True)
    merged_histories.update(all_histories)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged_histories, f, indent=2)
    print(f"Saved histories to {out_path}", flush=True)


def main() -> None:
    """Parse CLI arguments and train selected models with Lightning."""
    parser = argparse.ArgumentParser(description="Train chest X-ray pneumonia classifiers (Lightning).")
    parser.add_argument("--model", choices=["resnet18", "mobilenet", "swintiny", "all"], default="all")
    parser.add_argument("--train_dir", default=cfg.train_dir)
    parser.add_argument("--test_dir", default=cfg.test_dir)
    parser.add_argument("--val_dir", default=cfg.val_dir)
    parser.add_argument("--lr", type=float, default=None, help="CNN Adam LR (default cfg.learning_rate).")
    parser.add_argument("--mobilenet_lr", type=float, default=None)
    parser.add_argument("--resnet_lr", type=float, default=None)
    parser.add_argument("--swintiny_lr", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=None)
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use pre-exported .pt split caches instead of loading image folders directly.",
    )
    parser.add_argument("--cache_train_pt", type=str, default=None, help="Train split cache path.")
    parser.add_argument("--cache_val_pt", type=str, default=None, help="Validation split cache path.")
    parser.add_argument("--cache_test_pt", type=str, default=None, help="Test split cache path.")
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of GPUs to use (-1 = all visible CUDA devices, e.g. 2×T4 on Kaggle). Default: cfg.trainer_num_devices.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run key used in histories.json; also writes checkpoint to models/{run_name}_best.pth.",
    )
    args = parser.parse_args()
    _apply_cli_overrides(args)
    run_training(
        model=args.model,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        val_dir=args.val_dir,
        use_cache=args.use_cache,
        cache_train_pt=args.cache_train_pt,
        cache_val_pt=args.cache_val_pt,
        cache_test_pt=args.cache_test_pt,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
