
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np

try:
    import optuna
except ImportError as exc:
    raise RuntimeError(
        "pip install optuna!"
    ) from exc

import GNN as gnn

def _set_global_params(
    *,
    lr: float,
    weight_decay: float,
    lambda_data: float,
    lambda_pde: float,
    lambda_gauge: float,
    hidden_channels: int,
    num_layers: int,
    num_epochs: int,
    train_fraction: float,
    max_num_cases: int,
    random_seed: int,
    use_lr_warmup: bool = True,
    use_grad_clip: bool = True,
    pde_loss_normalization: str = "relative",
) -> None:

    gnn.LR = lr
    gnn.WEIGHT_DECAY = weight_decay
    gnn.LAMBDA_DATA = lambda_data
    gnn.LAMBDA_PDE = lambda_pde
    gnn.LAMBDA_GAUGE = lambda_gauge
    gnn.HIDDEN_CHANNELS = hidden_channels
    gnn.NUM_LAYERS = num_layers
    gnn.NUM_EPOCHS = num_epochs
    gnn.TRAIN_FRACTION = train_fraction
    gnn.MAX_NUM_CASES = max_num_cases
    gnn.RANDOM_SEED = random_seed
    gnn.USE_LR_WARMUP = use_lr_warmup
    gnn.USE_GRAD_CLIP = use_grad_clip
    gnn.PDE_LOSS_NORMALIZATION = pde_loss_normalization

def _initialize_log_file(log_file: Path) -> None:

    log_file.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# trial\tval_error\tdata_loss\tpde_loss\tlr\tweight_decay\tlambda_data\t"
        "lambda_pde\thidden_channels\tnum_layers\n"
    )

    if not log_file.exists():
        log_file.write_text(header, encoding="utf-8")

def _append_trial_result(
    log_file: Path,
    trial_number: int,
    val_error: float,
    data_loss: float,
    pde_loss: float,
    *,
    lr: float,
    weight_decay: float,
    lambda_data: float,
    lambda_pde: float,
    hidden_channels: int,
    num_layers: int,
) -> None:

    _initialize_log_file(log_file)
    line = (
        f"{trial_number}\t{val_error:.6e}\t{data_loss:.6e}\t{pde_loss:.6e}\t"
        f"{lr:.6e}\t{weight_decay:.6e}\t"
        f"{lambda_data:.6e}\t{lambda_pde:.6e}\t{hidden_channels}\t{num_layers}\n"
    )

    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)

def _extract_best_val_error(history: dict) -> tuple:

    val_errors = history["rel_err_val"]
    data_losses = history["data_loss"]
    pde_losses = history["pde_loss"]

    valid_indices = [i for i, v in enumerate(val_errors) if v is not None]

    if valid_indices:
        best_idx = min(valid_indices, key=lambda i: val_errors[i])
        return (
            float(val_errors[best_idx]),
            float(data_losses[best_idx]),
            float(pde_losses[best_idx]),
        )

    if history["rel_err_train"]:
        last_idx = len(history["rel_err_train"]) - 1
        return (
            float(history["rel_err_train"][last_idx]),
            float(data_losses[last_idx]),
            float(pde_losses[last_idx]),
        )

    raise RuntimeError("学習履歴が空のため評価指標を取得不可能")

def objective(
    trial: optuna.Trial,
    data_dir: str,
    num_epochs: int,
    max_num_cases: int,
    train_fraction: float,
    random_seed: int,
    log_file: Path,
    lambda_gauge: float,
    search_lambda_gauge: bool,
    lambda_data_min: float,
    lambda_data_max: float,
    lambda_pde_min: float,
    lambda_pde_max: float,
) -> float:

    lr = trial.suggest_float(name="lr", low=1e-4, high=1e-2, log=True)
    weight_decay = trial.suggest_float(name="weight_decay", low=1e-6, high=1e-3, log=True)

    if lambda_data_min == 0 and lambda_data_max == 0:
        lambda_data = 0.0
    elif lambda_data_min == lambda_data_max:
        lambda_data = lambda_data_min
    elif lambda_data_min > 0:
        lambda_data = trial.suggest_float(name="lambda_data", low=lambda_data_min, high=lambda_data_max, log=True)
    else:
        lambda_data = trial.suggest_float(name="lambda_data", low=lambda_data_min, high=lambda_data_max, log=False)

    if lambda_pde_min == 0 and lambda_pde_max == 0:
        lambda_pde = 0.0
    elif lambda_pde_min == lambda_pde_max:
        lambda_pde = lambda_pde_min
    elif lambda_pde_min > 0:
        lambda_pde = trial.suggest_float(name="lambda_pde", low=lambda_pde_min, high=lambda_pde_max, log=True)
    else:
        lambda_pde = trial.suggest_float(name="lambda_pde", low=lambda_pde_min, high=lambda_pde_max, log=False)

    hidden_channels = trial.suggest_int("hidden_channels", 32, 256, log=True)
    num_layers = trial.suggest_int("num_layers", 3, 7)

    if search_lambda_gauge:
        lambda_gauge_val = trial.suggest_float(
            name="lambda_gauge", low=1e-4, high=1.0, log=True
        )
    else:
        lambda_gauge_val = lambda_gauge

    _set_global_params(
        lr=lr,
        weight_decay=weight_decay,
        lambda_data=lambda_data,
        lambda_pde=lambda_pde,
        lambda_gauge=lambda_gauge_val,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_epochs=num_epochs,
        train_fraction=train_fraction,
        max_num_cases=max_num_cases,
        random_seed=random_seed,
    )

    random.seed(random_seed)
    np.random.seed(random_seed)

    history = gnn.train_gnn_auto_trainval_pde_weighted(
        data_dir,
        enable_plot=False,
        return_history=True,
    )

    val_error, data_loss, pde_loss = _extract_best_val_error(history)
    _append_trial_result(
        log_file,
        trial.number,
        val_error,
        data_loss,
        pde_loss,
        lr=lr,
        weight_decay=weight_decay,
        lambda_data=lambda_data,
        lambda_pde=lambda_pde,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
    )
    return val_error

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna によるハイパーパラメータ探索")
    parser.add_argument("--data_dir", default=gnn.DATA_DIR)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument(
        "--max_num_cases",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=Path("optuna_trials_history.tsv"),
    )
    parser.add_argument(
        "--lazy_loading",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_lazy_loading",
        action="store_true",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache",
    )
    parser.add_argument(
        "--lambda_gauge",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--search_lambda_gauge",
        action="store_true",
    )
    parser.add_argument(
        "--no_mesh_quality_weights",
        action="store_true",
    )
    parser.add_argument(
        "--no_diagonal_scaling",
        action="store_true",
    )
    parser.add_argument(
        "--lambda_data_min",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda_data_max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda_pde_min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lambda_pde_max",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--use_one_cycle_lr",
        action="store_true",
    )
    parser.add_argument(
        "--one_cycle_max_lr",
        type=float,
        default=1e-2,
    )

    args = parser.parse_args()

    gnn.USE_LAZY_LOADING = not args.no_lazy_loading
    gnn.USE_AMP = not args.no_amp

    gnn.USE_DATA_CACHE = not args.no_cache
    gnn.CACHE_DIR = args.cache_dir

    gnn.USE_MESH_QUALITY_WEIGHTS = not args.no_mesh_quality_weights

    gnn.USE_DIAGONAL_SCALING = not args.no_diagonal_scaling

    gnn.USE_EARLY_STOPPING = not args.no_early_stopping
    gnn.EARLY_STOPPING_PATIENCE = args.early_stopping_patience

    gnn.USE_ONE_CYCLE_LR = args.use_one_cycle_lr
    gnn.ONE_CYCLE_MAX_LR = args.one_cycle_max_lr
    if args.use_one_cycle_lr:
        gnn.USE_LR_SCHEDULER = False

    sampler = optuna.samplers.TPESampler(seed=args.random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    _initialize_log_file(args.log_file)

    if args.lambda_data_min == 0 and args.lambda_pde_min == 0 and args.lambda_data_max == 0 and args.lambda_pde_max == 0:
        raise ValueError(
            "lambda_data と lambda_pde が両方とも 0 に設定"
        )

    print(f"[INFO] データ損失の重みの範囲: [{args.lambda_data_min}, {args.lambda_data_max}]")
    print(f"[INFO] PDE 損失の重みの範囲: [{args.lambda_pde_min}, {args.lambda_pde_max}]")

    study.optimize(
        lambda trial: objective(
            trial,
            args.data_dir,
            args.num_epochs,
            args.max_num_cases,
            args.train_fraction,
            args.random_seed,
            args.log_file,
            args.lambda_gauge,
            args.search_lambda_gauge,
            args.lambda_data_min,
            args.lambda_data_max,
            args.lambda_pde_min,
            args.lambda_pde_max,
        ),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print(f"  最良のテスト誤差の値: {study.best_trial.value:.4e}")
    print("  ハイパーパラメータ:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    best = study.best_trial

    gnn.NUM_EPOCHS      = args.num_epochs
    gnn.MAX_NUM_CASES   = args.max_num_cases
    gnn.TRAIN_FRACTION  = args.train_fraction
    gnn.RANDOM_SEED     = args.random_seed

    gnn.LR              = best.params["lr"]
    gnn.WEIGHT_DECAY    = best.params["weight_decay"]
    gnn.LAMBDA_DATA     = best.params.get("lambda_data", args.lambda_data_min)
    gnn.LAMBDA_PDE      = best.params.get("lambda_pde", args.lambda_pde_min)
    gnn.HIDDEN_CHANNELS = best.params["hidden_channels"]
    gnn.NUM_LAYERS      = best.params["num_layers"]

    gnn.train_gnn_auto_trainval_pde_weighted(
        args.data_dir,
        enable_plot=False,
        return_history=False,
    )

if __name__ == "__main__":
    main()

