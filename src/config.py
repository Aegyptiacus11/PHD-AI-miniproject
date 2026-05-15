"""Global training and evaluation configuration."""

from dataclasses import dataclass, field


def _class_names() -> list[str]:
    """Binary labels for the chest X-ray pneumonia dataset."""
    return ["NORMAL", "PNEUMONIA"]


@dataclass
class Config:
    """Hyperparameters and paths for chest X-ray pneumonia training."""

    seed: int = 42
    num_classes: int = 2
    img_size: int = 128
    mobilenet_img_size: int = 224
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-3
    mobilenet_lr: float = 1e-4
    resnet_lr: float = 1e-4
    swintiny_lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.5
    freeze_backbone_epochs: int = 5
    class_names: list[str] = field(default_factory=_class_names)
    model_dir: str = "models"
    results_dir: str = "results"
    cnn_model_path: str = "models/cnn_best.pth"
    mobilenet_model_path: str = "models/mobilenet_best.pth"
    resnet18_model_path: str = "models/resnet18_best.pth"
    swintiny_model_path: str = "models/swintiny_best.pth"
    train_dir: str = "data/chest_xray/train"
    test_dir: str = "data/chest_xray/test"
    val_dir: str = "data/chest_xray/val"
    cache_train_pt: str = "data/cache/chest_xray_train.pt"
    cache_val_pt: str = "data/cache/chest_xray_val.pt"
    cache_test_pt: str = "data/cache/chest_xray_test.pt"
    histories_path: str = "results/histories.json"
    # Models compared in evaluate / default --model all order
    compare_models: tuple[str, ...] = ("resnet18", "mobilenet", "swintiny")
    transfer_model_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"mobilenet", "resnet18", "swintiny"})
    )
    # Lightning: ``-1`` = use every visible CUDA device (e.g. 2× Tesla T4 on Kaggle); ``1`` = single GPU
    trainer_num_devices: int = -1


cfg = Config()


def checkpoint_path_for_model(model_type: str) -> str:
    """Return the best-checkpoint path for a known ``model_type``."""
    if model_type == "cnn":
        return cfg.cnn_model_path
    if model_type == "mobilenet":
        return cfg.mobilenet_model_path
    if model_type == "resnet18":
        return cfg.resnet18_model_path
    if model_type == "swintiny":
        return cfg.swintiny_model_path
    raise ValueError(f"Unknown model_type for checkpoint: {model_type}")
