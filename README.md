# Chest X-Ray Pneumonia Classification

Mini-project comparing `resnet18`, `mobilenet`, and `swintiny` on the Kaggle Chest X-Ray Pneumonia dataset.

- Dataset: [paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Kaggle notebook: [nassimkaddouri/phd-ai-miniproject](https://www.kaggle.com/code/nassimkaddouri/phd-ai-miniproject)

## Quick start

```bash
uv sync
```

## Expected data layout

```text
data/chest_xray/
  train/{NORMAL,PNEUMONIA}
  test/{NORMAL,PNEUMONIA}
  val/{NORMAL,PNEUMONIA}   # optional
```

## Main commands

```bash
# train ResNet18
uv run python -m src.train --model resnet18 --train_dir data/chest_xray/train --test_dir data/chest_xray/test --val_dir data/chest_xray/val

# train MobileNetV2
uv run python -m src.train --model mobilenet --train_dir data/chest_xray/train --test_dir data/chest_xray/test --val_dir data/chest_xray/val

# train Swin-Tiny
uv run python -m src.train --model swintiny --train_dir data/chest_xray/train --test_dir data/chest_xray/test --val_dir data/chest_xray/val

# evaluate saved runs
uv run python -m src.evaluate --train_dir data/chest_xray/train --test_dir data/chest_xray/test --runs run_resnet18 run_mobilenet run_swintiny
```

## Outputs

- `results/`: training curves, confusion matrices, wrong prediction panels, and metric summaries (JSON/CSV/TEX).
