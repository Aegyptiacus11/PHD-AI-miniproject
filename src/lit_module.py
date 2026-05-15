"""PyTorch Lightning module for chest X-ray classifiers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import cfg
from src.model import ModelName, get_model


def _transfer_lr(model_type: ModelName) -> float:
    if model_type == "mobilenet":
        return cfg.mobilenet_lr
    if model_type == "resnet18":
        return cfg.resnet_lr
    if model_type == "swintiny":
        return cfg.swintiny_lr
    return cfg.learning_rate


class LitSignClassifier(L.LightningModule):
    """Single Lightning module for CNN, MobileNetV2, and ResNet18."""

    def __init__(
        self,
        model_type: ModelName,
        *,
        freeze_backbone: bool,
        max_epochs_this_phase: int,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.freeze_backbone = freeze_backbone
        self.max_epochs_this_phase = max_epochs_this_phase
        self.save_hyperparameters()
        self.net = get_model(model_type, torch.device("cpu"), freeze_backbone=freeze_backbone)
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_time": [],
        }
        self._train_loss_sum = 0.0
        self._train_n = 0
        self._train_correct = 0
        self._val_loss_sum = 0.0
        self._val_n = 0
        self._val_correct = 0
        self._epoch_t0 = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_epoch_start(self) -> None:
        self._epoch_t0 = time.perf_counter()
        self._train_loss_sum = 0.0
        self._train_n = 0
        self._train_correct = 0

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        self._train_loss_sum += float(loss.detach()) * bs
        self._train_n += bs
        self._train_correct += int((logits.argmax(dim=1) == y).sum().item())
        self.log("train_loss_step", loss, prog_bar=False)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if self._train_n == 0:
            return
        tl = self._train_loss_sum / self._train_n
        ta = self._train_correct / self._train_n
        self.history["train_loss"].append(float(tl))
        self.history["train_acc"].append(float(ta))
        self.log("train_loss", tl, prog_bar=True)
        self.log("train_acc", ta, prog_bar=True)
        self.history["epoch_time"].append(float(time.perf_counter() - self._epoch_t0))

    def on_validation_epoch_start(self) -> None:
        self._val_loss_sum = 0.0
        self._val_n = 0
        self._val_correct = 0

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        self._val_loss_sum += float(loss.detach()) * bs
        self._val_n += bs
        self._val_correct += int((logits.argmax(dim=1) == y).sum().item())

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        if self._val_n == 0:
            return
        vl = self._val_loss_sum / self._val_n
        va = self._val_correct / self._val_n
        self.history["val_loss"].append(float(vl))
        self.history["val_acc"].append(float(va))
        self.log("val_loss", vl, prog_bar=True)
        self.log("val_acc", va, prog_bar=True)

    def configure_optimizers(self) -> Any:
        if self.model_type == "cnn":
            opt = Adam(self.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            sched = CosineAnnealingLR(opt, T_max=self.max_epochs_this_phase)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        lr = _transfer_lr(self.model_type)
        if self.freeze_backbone:
            params = [p for p in self.net.parameters() if p.requires_grad]
            opt = Adam(params, lr=lr, weight_decay=cfg.weight_decay)
        else:
            opt = Adam(self.net.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.max_epochs_this_phase)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


class BestValCheckpointCallback(Callback):
    """Save ``{model_state, epoch, val_acc}`` when validation accuracy improves."""

    def __init__(self, checkpoint_path: str | Path, model_type: str, epoch_offset: int = 0) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.epoch_offset = epoch_offset
        self.best_val: float = -1.0
        self.best_epoch: int = -1

    def on_validation_end(self, trainer: L.Trainer, pl_module: LitSignClassifier) -> None:
        if trainer.sanity_checking:
            return
        if trainer.global_rank != 0:
            return
        if not pl_module.history["val_acc"]:
            return
        va = float(pl_module.history["val_acc"][-1])
        ep_global = self.epoch_offset + trainer.current_epoch
        if va > self.best_val:
            self.best_val = va
            self.best_epoch = ep_global
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": pl_module.net.state_dict(), "epoch": ep_global, "val_acc": va},
                self.checkpoint_path,
            )
            print(
                f"[{self.model_type}] saved best val_acc={va:.4f} epoch={ep_global + 1} -> {self.checkpoint_path}",
                flush=True,
            )
