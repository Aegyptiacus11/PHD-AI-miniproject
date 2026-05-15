"""Backbone models and factory for chest X-ray classification."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    Swin_T_Weights,
    mobilenet_v2,
    resnet18,
    swin_t,
)

from src.config import cfg

ModelName = Literal["cnn", "mobilenet", "resnet18", "swintiny"]


@runtime_checkable
class BackboneFreezable(Protocol):
    """Models that support freezing ImageNet backbones for phased fine-tuning."""

    def freeze_backbone(self) -> None: ...

    def unfreeze_backbone(self) -> None: ...


class CustomCNN(nn.Module):
    """Lightweight CNN for grayscale chest X-ray images."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 with ImageNet weights and a custom classification head."""

    def __init__(self, num_classes: int = 24, dropout: float = 0.5) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


class ResNet18Classifier(nn.Module):
    """ResNet18 with ImageNet weights and a replaced classification head."""

    def __init__(self, num_classes: int = 24, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_f = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    def freeze_backbone(self) -> None:
        for name, param in self.net.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.net.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwinTinyClassifier(nn.Module):
    """Swin-T with ImageNet weights and replaced classification head."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_f = self.net.head.in_features
        self.net.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))

    def freeze_backbone(self) -> None:
        for name, param in self.net.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.net.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model(
    model_type: ModelName,
    device: torch.device,
    *,
    freeze_backbone: bool | None = None,
) -> nn.Module:
    """
    Instantiate the requested model and move it to ``device``.

    Args:
        model_type: ``cnn``, ``mobilenet``, ``resnet18``, or ``swintiny``.
        device: Target device for parameters and buffers.
        freeze_backbone: For transfer models, whether to start with the backbone frozen.
            Default ``True`` for transfer models; ignored for ``cnn``.
    """
    if model_type == "cnn":
        model: nn.Module = CustomCNN(num_classes=cfg.num_classes, dropout=cfg.dropout)
        return model.to(device)

    if freeze_backbone is None:
        freeze_backbone = True

    if model_type == "mobilenet":
        model = MobileNetV2Classifier(num_classes=cfg.num_classes, dropout=cfg.dropout)
    elif model_type == "resnet18":
        model = ResNet18Classifier(num_classes=cfg.num_classes, dropout=cfg.dropout)
    elif model_type == "swintiny":
        model = SwinTinyClassifier(num_classes=cfg.num_classes, dropout=cfg.dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if freeze_backbone:
        assert isinstance(model, BackboneFreezable)
        model.freeze_backbone()
    else:
        assert isinstance(model, BackboneFreezable)
        model.unfreeze_backbone()

    return model.to(device)
