"""Evaluation utilities, plots, and CLI for trained models."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.config import cfg, checkpoint_path_for_model
from src.dataset import get_dataloaders
from src.model import get_model
from src.train import set_seed

ModelName = str


def get_predictions(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Collect predictions, labels, and input tensors for a loader.

    Returns:
        ``(preds, labels, images)`` as NumPy arrays for preds/labels and a CPU tensor for images.
    """
    model.eval()
    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    images_list: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs_dev = inputs.to(device, non_blocking=True)
            logits = model(inputs_dev)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds_list.append(pred)
            labels_list.append(targets.numpy())
            images_list.append(inputs.detach().cpu())
    preds = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    images = torch.cat(images_list, dim=0)
    return preds, labels, images


def _strip_history(entry: dict[str, Any]) -> dict[str, list[float]]:
    keys = {"train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"}
    return {k: entry[k] for k in keys if k in entry}


def plot_curves(histories: dict[str, dict[str, Any]]) -> None:
    """
    Plot side-by-side accuracy and loss curves for all models on one figure.

    Saves ``results/training_curves.png`` with ``dpi=150``.
    """
    out = Path(cfg.results_dir) / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    for name, data in histories.items():
        h = _strip_history(data)
        if not h.get("train_acc"):
            continue
        n = min(len(h["train_acc"]), len(h["val_acc"]), len(h["train_loss"]), len(h["val_loss"]))
        if n == 0:
            continue
        epochs = np.arange(1, n + 1)
        axes[0].plot(epochs, h["train_acc"][:n], label=f"{name} train")
        axes[0].plot(epochs, h["val_acc"][:n], linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, h["train_loss"][:n], label=f"{name} train")
        axes[1].plot(epochs, h["val_loss"][:n], linestyle="--", label=f"{name} val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.show()


def plot_curves_per_model(histories: dict[str, dict[str, Any]]) -> None:
    """Plot individual train/val curves for each model as separate figures.

    A vertical dashed line marks the freeze-to-unfreeze boundary for transfer models.
    """
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_epoch = cfg.freeze_backbone_epochs
    for name, data in histories.items():
        h = _strip_history(data)
        if not h.get("train_acc"):
            continue
        n = min(len(h["train_acc"]), len(h["val_acc"]), len(h["train_loss"]), len(h["val_loss"]))
        if n == 0:
            continue
        model_type = data.get("model_type", "")
        is_transfer = model_type in cfg.transfer_model_types
        epochs = np.arange(1, n + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        axes[0].plot(epochs, h["train_acc"][:n], label="Train")
        axes[0].plot(epochs, h["val_acc"][:n], linestyle="--", label="Val")
        if is_transfer and freeze_epoch < n:
            axes[0].axvline(x=freeze_epoch + 0.5, color="gray", linestyle=":", alpha=0.7, label="Unfreeze")
        axes[0].set_title(f"{name} — Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, h["train_loss"][:n], label="Train")
        axes[1].plot(epochs, h["val_loss"][:n], linestyle="--", label="Val")
        if is_transfer and freeze_epoch < n:
            axes[1].axvline(x=freeze_epoch + 0.5, color="gray", linestyle=":", alpha=0.7, label="Unfreeze")
        axes[1].set_title(f"{name} — Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"training_curves_{name}.png"
        fig.savefig(out)
        plt.show()
        print(f"Saved per-model curves to {out}")


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, model_type: ModelName, run_key: str) -> None:
    """
    Plot a row-normalized confusion matrix (color by % of true class) with raw counts annotated.

    Saves ``results/confusion_matrix_{model_type}.png``.
    """
    cm = confusion_matrix(labels, preds, labels=np.arange(cfg.num_classes))
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, np.maximum(row_sums, 1)) * 100.0
    out = Path(cfg.results_dir) / f"confusion_matrix_{run_key}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(
        pct,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=cfg.class_names,
        yticklabels=cfg.class_names,
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Confusion matrix ({run_key} / {model_type}) — color: % of true row; cells: raw counts"
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.show()


def _to_display_gray(images: torch.Tensor, model_type: ModelName) -> np.ndarray:
    """Convert normalized tensors to ``H x W`` float arrays in ``[0, 1]`` for imshow."""
    x = images.clone()
    if model_type == "cnn":
        x = x * 0.5 + 0.5
        return x.squeeze(1).numpy().clip(0.0, 1.0)
    # Transfer models (MobileNet, ResNet): ImageNet normalization, show as grayscale average
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = x * std + mean
    gray = x.mean(dim=1).numpy().clip(0.0, 1.0)
    return gray


def plot_wrong_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    images: torch.Tensor,
    model_type: ModelName,
    run_key: str,
) -> None:
    """
    Plot a 4x4 grid of misclassified examples with true vs predicted letter names.

    Saves ``results/wrong_predictions_{model_type}.png``.
    """
    wrong_idx = np.where(preds != labels)[0]
    if wrong_idx.size == 0:
        print(f"[{model_type}] no misclassified samples to plot.")
        return
    pick = wrong_idx[:16]
    grid = _to_display_gray(images[pick], model_type)
    out = Path(cfg.results_dir) / f"wrong_predictions_{run_key}.png"
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), dpi=150)
    for ax, idx, g in zip(axes.ravel(), pick, grid):
        ax.imshow(g, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
        tname = cfg.class_names[int(labels[idx])]
        pname = cfg.class_names[int(preds[idx])]
        ax.set_title(f"T:{tname} P:{pname}", fontsize=8)
    fig.suptitle(f"Misclassified samples ({run_key} / {model_type})")
    fig.tight_layout()
    fig.savefig(out)
    plt.show()


def save_classification_report(
    preds: np.ndarray, labels: np.ndarray, model_type: ModelName, run_key: str
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Write sklearn classification report and return per-class + overall metrics."""
    out = Path(cfg.results_dir) / f"classification_report_{run_key}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        labels,
        preds,
        labels=np.arange(cfg.num_classes),
        target_names=cfg.class_names,
        digits=4,
        zero_division=0,
    )
    report_dict: dict[str, Any] = classification_report(
        labels,
        preds,
        labels=np.arange(cfg.num_classes),
        target_names=cfg.class_names,
        output_dict=True,
        zero_division=0,
    )
    out.write_text(report + "\n", encoding="utf-8")
    print(f"Saved classification report to {out}")
    per_class: dict[str, dict[str, float]] = {}
    for cls_name in cfg.class_names:
        cls_row = report_dict.get(cls_name, {})
        per_class[cls_name] = {
            "precision": float(cls_row.get("precision", 0.0)),
            "recall": float(cls_row.get("recall", 0.0)),
            "f1": float(cls_row.get("f1-score", 0.0)),
            "support": float(cls_row.get("support", 0.0)),
        }
    macro = report_dict.get("macro avg", {})
    weighted = report_dict.get("weighted avg", {})
    overall = {
        "accuracy": float(report_dict.get("accuracy", 0.0)),
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1-score", 0.0)),
        "weighted_precision": float(weighted.get("precision", 0.0)),
        "weighted_recall": float(weighted.get("recall", 0.0)),
        "weighted_f1": float(weighted.get("f1-score", 0.0)),
    }
    return per_class, overall


def save_model_comparison_summary(rows: list[tuple[str, float, int, float]]) -> None:
    """Save model/test summary (acc + params + train_time) to JSON/CSV/LaTeX."""
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "model_comparison_summary.json"
    csv_out = out_dir / "model_comparison_summary.csv"
    tex_out = out_dir / "model_comparison_summary.tex"

    payload = [
        {"model": m, "test_acc": acc, "params": params, "train_time_s": t}
        for m, acc, params, t in rows
    ]
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_lines = ["model,test_acc,params,train_time_s"]
    for m, acc, params, t in rows:
        csv_lines.append(f"{m},{acc:.4f},{params},{t:.1f}")
    csv_out.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Test Acc & Params & Train Time (s) \\",
        r"\midrule",
    ]
    for m, acc, params, t in rows:
        tex_lines.append(f"{m} & {acc:.4f} & {params:,} & {t:.1f} " + r"\\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_out.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Saved model comparison summary to {json_out}, {csv_out}, {tex_out}")


def save_overall_metrics_summary(
    rows: list[tuple[str, float, float, float, float, float, float, float]],
) -> None:
    """Save overall test metrics (accuracy + macro/weighted P/R/F1) to JSON/CSV/LaTeX."""
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "overall_metrics_summary.json"
    csv_out = out_dir / "overall_metrics_summary.csv"
    tex_out = out_dir / "overall_metrics_summary.tex"

    payload = [
        {
            "model": m,
            "accuracy": acc,
            "macro_precision": mp,
            "macro_recall": mr,
            "macro_f1": mf1,
            "weighted_precision": wp,
            "weighted_recall": wr,
            "weighted_f1": wf1,
        }
        for m, acc, mp, mr, mf1, wp, wr, wf1 in rows
    ]
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_lines = [
        "model,accuracy,macro_precision,macro_recall,macro_f1,weighted_precision,weighted_recall,weighted_f1"
    ]
    for m, acc, mp, mr, mf1, wp, wr, wf1 in rows:
        csv_lines.append(f"{m},{acc:.4f},{mp:.4f},{mr:.4f},{mf1:.4f},{wp:.4f},{wr:.4f},{wf1:.4f}")
    csv_out.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & Acc & Macro P & Macro R & Macro F1 & Weighted P & Weighted R & Weighted F1 \\",
        r"\midrule",
    ]
    for m, acc, mp, mr, mf1, wp, wr, wf1 in rows:
        tex_lines.append(
            f"{m} & {acc:.4f} & {mp:.4f} & {mr:.4f} & {mf1:.4f} & {wp:.4f} & {wr:.4f} & {wf1:.4f} " + r"\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_out.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Saved overall metrics summary to {json_out}, {csv_out}, {tex_out}")


def save_per_class_summary(
    rows: list[tuple[str, str, float, float, float, float]],
) -> None:
    """Persist per-class precision/recall/F1/support as JSON, CSV, and LaTeX tabular."""
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "per_class_metrics_summary.json"
    csv_out = out_dir / "per_class_metrics_summary.csv"
    tex_out = out_dir / "per_class_metrics_summary.tex"

    payload = [
        {
            "model": m,
            "class": c,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": s,
        }
        for m, c, p, r, f1, s in rows
    ]
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_lines = ["model,class,precision,recall,f1,support"]
    for m, c, p, r, f1, s in rows:
        csv_lines.append(f"{m},{c},{p:.4f},{r:.4f},{f1:.4f},{int(s)}")
    csv_out.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Class & Precision & Recall & F1 & Support \\",
        r"\midrule",
    ]
    prev_model: str | None = None
    for m, c, p, r, f1, s in rows:
        if prev_model is not None and m != prev_model:
            tex_lines.append(r"\midrule")
        tex_lines.append(f"{m} & {c} & {p:.4f} & {r:.4f} & {f1:.4f} & {int(s)} " + r"\\")
        prev_model = m
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_out.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Saved per-class summary to {json_out}, {csv_out}, {tex_out}")


def copy_results_to_report_figures() -> None:
    """Copy report-relevant result artifacts to ``report/figures``."""
    src = Path(cfg.results_dir)
    dst = Path("report/figures")
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in ("*.png", "*.txt", "*.csv", "*.json", "*.tex"):
        for p in src.glob(pattern):
            if p.is_file():
                shutil.copy2(p, dst / p.name)
                copied += 1
    print(f"Copied {copied} result files to {dst}")


def _load_weights(model: nn.Module, path: Path, model_type: ModelName) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if hasattr(model, "unfreeze_backbone"):
        model.unfreeze_backbone()
    model.load_state_dict(state)


def load_model_for_inference(
    model_type: Literal["mobilenet", "resnet18", "swintiny"], device: torch.device
) -> nn.Module:
    """
    Build a model, load the best checkpoint from ``cfg``, and move it to ``device``.

    Args:
        model_type: Which architecture to restore.
        device: Target device.

    Returns:
        Model in evaluation mode (caller may still call ``.eval()`` explicitly).
    """
    path = Path(checkpoint_path_for_model(model_type))
    model = get_model(model_type, device, freeze_backbone=False)
    _load_weights(model, path, model_type)
    return model.to(device)


def _apply_eval_cli(args: argparse.Namespace) -> None:
    if args.train_dir is not None:
        cfg.train_dir = args.train_dir
    if args.test_dir is not None:
        cfg.test_dir = args.test_dir
    if args.val_dir is not None:
        cfg.val_dir = args.val_dir
    if args.histories_path is not None:
        cfg.histories_path = args.histories_path
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    if args.cache_train_pt is not None:
        cfg.cache_train_pt = args.cache_train_pt
    if args.cache_val_pt is not None:
        cfg.cache_val_pt = args.cache_val_pt
    if args.cache_test_pt is not None:
        cfg.cache_test_pt = args.cache_test_pt


def run_evaluation(
    *,
    train_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    val_dir: str | Path | None = None,
    histories_path: str | Path | None = None,
    results_dir: str | None = None,
    use_cache: bool = False,
    cache_train_pt: str | None = None,
    cache_val_pt: str | None = None,
    cache_test_pt: str | None = None,
    models: list[str] | None = None,
    runs: list[str] | None = None,
) -> None:
    """Programmatic entry point (e.g. notebooks): same behavior as the ``src.evaluate`` CLI."""
    if train_dir is not None:
        cfg.train_dir = str(train_dir)
    if test_dir is not None:
        cfg.test_dir = str(test_dir)
    if val_dir is not None:
        cfg.val_dir = str(val_dir)
    if histories_path is not None:
        cfg.histories_path = str(histories_path)
    if results_dir is not None:
        cfg.results_dir = results_dir
    if cache_train_pt is not None:
        cfg.cache_train_pt = cache_train_pt
    if cache_val_pt is not None:
        cfg.cache_val_pt = cache_val_pt
    if cache_test_pt is not None:
        cfg.cache_test_pt = cache_test_pt

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = Path(cfg.histories_path)
    if not hp.is_file():
        raise FileNotFoundError(f"Missing histories file: {hp}. Run training first.")
    histories: dict[str, dict[str, Any]] = json.loads(hp.read_text(encoding="utf-8"))

    summary_rows: list[tuple[str, float, int, float]] = []
    overall_rows: list[tuple[str, float, float, float, float, float, float, float]] = []
    per_class_rows: list[tuple[str, str, float, float, float, float]] = []
    if runs is not None and models is not None:
        raise ValueError("Use either `runs` or `models`, not both.")

    if runs is not None:
        missing_runs = [r for r in runs if r not in histories]
        if missing_runs:
            raise ValueError(f"Unknown run(s) in histories: {missing_runs}")
        eval_plan: list[tuple[str, str, Path]] = []
        for run_key in runs:
            row = histories.get(run_key, {})
            model_type = str(row.get("model_type", ""))
            if model_type not in cfg.compare_models:
                raise ValueError(
                    f"Run `{run_key}` is missing a valid `model_type` in histories.json "
                    f"(found `{model_type}`)."
                )
            ckpt_raw = row.get("checkpoint_path")
            ckpt_path = Path(str(ckpt_raw)) if ckpt_raw else Path(checkpoint_path_for_model(model_type))
            eval_plan.append((run_key, model_type, ckpt_path))
    else:
        eval_models = list(models) if models is not None else list(cfg.compare_models)
        unknown = [m for m in eval_models if m not in cfg.compare_models]
        if unknown:
            raise ValueError(f"Unknown model(s) for evaluation: {unknown}. Valid: {list(cfg.compare_models)}")
        eval_plan = [(m, m, Path(checkpoint_path_for_model(m))) for m in eval_models]

    for run_key, model_type, ckpt_path in eval_plan:
        if not ckpt_path.is_file():
            print(f"Skipping {run_key}: missing checkpoint {ckpt_path}")
            continue
        _, test_loader = get_dataloaders(
            cfg.train_dir,
            cfg.test_dir,
            model_type=model_type,  # type: ignore[arg-type]
            val_dir=None,
            use_cache=use_cache,
            cache_train_pt=cfg.cache_train_pt,
            cache_val_pt=cfg.cache_val_pt,
            cache_test_pt=cfg.cache_test_pt,
        )
        model = get_model(model_type, device, freeze_backbone=False)  # type: ignore[arg-type]
        _load_weights(model, ckpt_path, model_type)
        model = model.to(device)
        preds, labels, images = get_predictions(model, test_loader, device)
        acc = float((preds == labels).mean())
        n_params = sum(p.numel() for p in model.parameters())
        total_time = float(sum(histories.get(run_key, {}).get("epoch_time", []) or []))
        summary_rows.append((f"{run_key}({model_type})", acc, n_params, total_time))

        plot_confusion_matrix(preds, labels, model_type, run_key)
        plot_wrong_predictions(preds, labels, images, model_type, run_key)
        per_class, overall = save_classification_report(preds, labels, model_type, run_key)
        overall_rows.append(
            (
                f"{run_key}({model_type})",
                float(overall.get("accuracy", acc)),
                float(overall.get("macro_precision", 0.0)),
                float(overall.get("macro_recall", 0.0)),
                float(overall.get("macro_f1", 0.0)),
                float(overall.get("weighted_precision", 0.0)),
                float(overall.get("weighted_recall", 0.0)),
                float(overall.get("weighted_f1", 0.0)),
            )
        )
        for cls_name in cfg.class_names:
            cls_metrics = per_class.get(cls_name, {})
            per_class_rows.append(
                (
                    f"{run_key}({model_type})",
                    cls_name,
                    float(cls_metrics.get("precision", 0.0)),
                    float(cls_metrics.get("recall", 0.0)),
                    float(cls_metrics.get("f1", 0.0)),
                    float(cls_metrics.get("support", 0.0)),
                )
            )

    selected_histories = {rk: histories.get(rk, {}) for rk, _, _ in eval_plan}
    plot_curves(selected_histories)
    plot_curves_per_model(selected_histories)

    print("\nModel comparison (test set):")
    print(f"{'model':<12}{'test_acc':>12}{'params':>14}{'train_time_s':>14}")
    for name, acc, n_params, ttime in summary_rows:
        print(f"{name:<12}{acc:12.4f}{n_params:14d}{ttime:14.1f}")
    if summary_rows:
        save_model_comparison_summary(summary_rows)
    if overall_rows:
        save_overall_metrics_summary(overall_rows)
    if per_class_rows:
        save_per_class_summary(per_class_rows)
    copy_results_to_report_figures()


def main() -> None:
    """Load histories, evaluate checkpoints, and generate all plots and reports."""
    parser = argparse.ArgumentParser(description="Evaluate saved chest X-ray checkpoints.")
    parser.add_argument("--train_dir", default=None, help="Override cfg.train_dir for dataloaders.")
    parser.add_argument("--test_dir", default=None, help="Override cfg.test_dir for dataloaders.")
    parser.add_argument("--val_dir", default=None, help="Optional validation directory to evaluate on.")
    parser.add_argument("--histories_path", default=None, help="Override path to results/histories.json.")
    parser.add_argument("--results_dir", default=None, help="Override directory for plots and reports.")
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use pre-exported .pt split caches instead of loading image folders directly.",
    )
    parser.add_argument("--cache_train_pt", default=None, help="Train split cache path.")
    parser.add_argument("--cache_val_pt", default=None, help="Validation split cache path.")
    parser.add_argument("--cache_test_pt", default=None, help="Test split cache path.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(cfg.compare_models),
        help="Explicit list of models to evaluate (default: cfg.compare_models).",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Explicit run keys from results/histories.json; evaluates each run's best checkpoint.",
    )
    args = parser.parse_args()
    _apply_eval_cli(args)
    run_evaluation(
        use_cache=args.use_cache,
        cache_train_pt=args.cache_train_pt,
        cache_val_pt=args.cache_val_pt,
        cache_test_pt=args.cache_test_pt,
        models=args.models,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
