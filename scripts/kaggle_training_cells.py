# Kaggle / local quick cells using uv CLI commands.
# Run each `# %%` block in Cursor/VS Code.

# %%
import os
import subprocess
from pathlib import Path

REPO_URL = os.environ.get("KAGGLE_REPO_URL", "").strip() or "https://github.com/Aegyptiacus11/PHD-AI-miniproject.git"
REPO_DIR = Path("/kaggle/working/PHD-AI-miniproject")


def _run(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


if Path("/kaggle").is_dir():
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    if (REPO_DIR / "requirements.txt").is_file():
        print("Repo already present at", REPO_DIR, "— skipping clone.")
    else:
        _run(f'git clone --depth 1 "{REPO_URL}" "{REPO_DIR}"')
    os.chdir(REPO_DIR)
    _run("pip install -q uv")
    _run("UV_LINK_MODE=copy uv sync")
    print("Kaggle setup OK · cwd =", Path.cwd())
else:
    print("Not on Kaggle — skip clone/install. Use `uv sync` and run from repo root.")

# %%
import os
from pathlib import Path

from src.kaggle_input import resolve_chest_xray_dirs

REPO_ROOT = Path.cwd().resolve()
os.chdir(REPO_ROOT)
TRAIN_DIR, TEST_DIR, VAL_DIR = resolve_chest_xray_dirs()
RUN_RESNET18 = "run_resnet18"
RUN_MOBILENET = "run_mobilenet"
RUN_VIT = "run_vit"
RUN_SWINTINY = "run_swintiny"
CACHE_TRAIN = "data/cache/chest_xray_train.pt"
CACHE_VAL = "data/cache/chest_xray_val.pt"
CACHE_TEST = "data/cache/chest_xray_test.pt"
USE_TENSOR_CACHE = os.environ.get("USE_TENSOR_CACHE", "1").strip() == "1"
VAL_ARG = f' --val_dir "{VAL_DIR}"' if VAL_DIR else ""
CACHE_ARGS = (
    f' --use_cache --cache_train_pt "{CACHE_TRAIN}" --cache_val_pt "{CACHE_VAL}" --cache_test_pt "{CACHE_TEST}"'
    if USE_TENSOR_CACHE
    else ""
)

print("REPO_ROOT:", REPO_ROOT)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("VAL_DIR:", VAL_DIR)
print("USE_TENSOR_CACHE:", USE_TENSOR_CACHE)

# %%
os.system("mkdir -p data/cache models results")
if USE_TENSOR_CACHE:
    os.system(
        f'uv run python -m src.export_cache --train_dir "{TRAIN_DIR}" --test_dir "{TEST_DIR}"'
        f'{VAL_ARG} --out_train "{CACHE_TRAIN}" --out_val "{CACHE_VAL}" --out_test "{CACHE_TEST}" --image_size 224'
    )

# %%
os.system(
    f'uv run python -m src.preview --train_dir "{TRAIN_DIR}" --test_dir "{TEST_DIR}" '
    f'{VAL_ARG} --out results/notebook_sample_grid.png --n 16 --dpi 150 '
    f'--problem_statement_dir report/figures'
)

# %%
os.system(
    f'uv run python -m src.train --model resnet18 --run_name "{RUN_RESNET18}" --train_dir "{TRAIN_DIR}" '
    f'--test_dir "{TEST_DIR}"{VAL_ARG}{CACHE_ARGS} '
    '--mobilenet_lr 1e-4 --resnet_lr 1e-4 --vit_lr 1e-4 --swintiny_lr 1e-4 '
    '--num_epochs 20 --batch_size 64 --weight_decay 1e-4 --seed 42 '
    '--freeze_backbone_epochs 5 --devices -1'
)

# %%
os.system(
    f'uv run python -m src.train --model mobilenet --run_name "{RUN_MOBILENET}" --train_dir "{TRAIN_DIR}" '
    f'--test_dir "{TEST_DIR}"{VAL_ARG}{CACHE_ARGS} '
    '--mobilenet_lr 1e-4 --resnet_lr 1e-4 --vit_lr 1e-4 --swintiny_lr 1e-4 '
    '--num_epochs 20 --batch_size 64 --weight_decay 1e-4 --seed 42 '
    '--freeze_backbone_epochs 5 --devices -1'
)

# %%
os.system(
    f'uv run python -m src.train --model vit --run_name "{RUN_VIT}" --train_dir "{TRAIN_DIR}" '
    f'--test_dir "{TEST_DIR}"{VAL_ARG}{CACHE_ARGS} '
    '--mobilenet_lr 1e-4 --resnet_lr 1e-4 --vit_lr 1e-4 --swintiny_lr 1e-4 '
    '--num_epochs 20 --batch_size 64 --weight_decay 1e-4 --seed 42 '
    '--freeze_backbone_epochs 5 --devices -1'
)

# %%
os.system(
    f'uv run python -m src.train --model swintiny --run_name "{RUN_SWINTINY}" --train_dir "{TRAIN_DIR}" '
    f'--test_dir "{TEST_DIR}"{VAL_ARG}{CACHE_ARGS} '
    '--mobilenet_lr 1e-4 --resnet_lr 1e-4 --vit_lr 1e-4 --swintiny_lr 1e-4 '
    '--num_epochs 20 --batch_size 64 --weight_decay 1e-4 --seed 42 '
    '--freeze_backbone_epochs 5 --devices -1'
)

# %%
os.system(
    f'uv run python -m src.evaluate --train_dir "{TRAIN_DIR}" --test_dir "{TEST_DIR}" '
    f'{CACHE_ARGS} --runs "{RUN_RESNET18}" "{RUN_MOBILENET}" "{RUN_VIT}" "{RUN_SWINTINY}"'
)
