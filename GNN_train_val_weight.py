#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime
import pickle
import hashlib
from scipy.sparse import csr_matrix
plt.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['axes.unicode_minus'] = False

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    raise RuntimeError(
        "pip install torch-geometric が必要"
    )

DATA_DIR       = "./data"
OUTPUT_DIR     = "./"
NUM_EPOCHS     = 1000
LR             = 1e-3
WEIGHT_DECAY   = 1e-5
MAX_NUM_CASES  = 100
TRAIN_FRACTION = 0.8
HIDDEN_CHANNELS = 64
NUM_LAYERS      = 4

USE_LR_SCHEDULER = True
LR_SCHED_FACTOR = 0.5
LR_SCHED_PATIENCE = 20
LR_SCHED_MIN_LR = 1e-6

USE_ONE_CYCLE_LR = False
ONE_CYCLE_MAX_LR = 1e-2
ONE_CYCLE_PCT_START = 0.3

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 1e-6

USE_LR_WARMUP = True
LR_WARMUP_EPOCHS = 10

USE_GRAD_CLIP = True
GRAD_CLIP_MAX_NORM = 1.0

USE_LAZY_LOADING = True
USE_AMP = True

USE_DATA_CACHE = True
CACHE_DIR = ".cache"

LAMBDA_DATA = 0.1
LAMBDA_PDE  = 0.01
LAMBDA_GAUGE = 0.01

W_PDE_MAX = 10.0
USE_MESH_QUALITY_WEIGHTS = True
USE_DIAGONAL_SCALING = True
PDE_LOSS_NORMALIZATION = "relative"

EPS      = 1.0e-12
EPS_DATA = 1e-12
EPS_RES  = 1e-8
EPS_PLOT = 1e-12

RANDOM_SEED = 42

PLOT_INTERVAL = 10

LOGGER_FILE = None

def log_print(msg: str):
    print(msg)
    global LOGGER_FILE
    if LOGGER_FILE is not None:
        print(msg, file=LOGGER_FILE)
        LOGGER_FILE.flush()

import re
import glob

def find_time_rank_list(data_dir: str):
    time_rank_tuples = []
    pattern = re.compile(r"^pEqn_(.+)_rank(\d+)\.dat$")

    missing_pEqn = []
    missing_csr = []
    missing_x = []

    gnn_dirs = glob.glob(os.path.join(data_dir, "processor*", "gnn"))

    if not gnn_dirs:
        return [], {"no_gnn_dirs": True}

    for gnn_dir in gnn_dirs:
        if not os.path.isdir(gnn_dir):
            continue

        for fn in os.listdir(gnn_dir):
            match = pattern.match(fn)
            if not match:
                continue

            time_str = match.group(1)
            rank_str = match.group(2)

            x_path   = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")
            csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
            csr_path_with_rank = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

            has_csr = os.path.exists(csr_path) or os.path.exists(csr_path_with_rank)
            has_x = os.path.exists(x_path)

            if has_csr:
                time_rank_tuples.append((time_str, rank_str, gnn_dir))
                if not has_x:
                    missing_x.append(x_path)
            else:
                missing_csr.append(csr_path)

    time_rank_tuples = sorted(
        set(time_rank_tuples),
        key=lambda tr: (float(tr[0]), int(tr[1]))
    )

    missing_info = {
        "missing_pEqn": missing_pEqn,
        "missing_csr": missing_csr,
        "missing_x": missing_x,
    }

    return time_rank_tuples, missing_info

def _compute_cache_key(data_dir: str, time_rank_tuples: list) -> str:
    key_str = data_dir + "|" + str(sorted(time_rank_tuples))
    return hashlib.md5(key_str.encode()).hexdigest()[:16]

def _get_cache_path(data_dir: str, time_rank_tuples: list) -> str:
    cache_key = _compute_cache_key(data_dir, time_rank_tuples)
    return os.path.join(CACHE_DIR, f"raw_cases_{cache_key}.pkl")

def _is_cache_valid(cache_path: str, time_rank_tuples: list) -> bool:
    if not os.path.exists(cache_path):
        return False

    cache_mtime = os.path.getmtime(cache_path)

    for time_str, rank_str, gnn_dir in time_rank_tuples:
        p_path = os.path.join(gnn_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
        x_path = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")
        csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
        csr_path_with_rank = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

        for path in [p_path, x_path]:
            if os.path.exists(path) and os.path.getmtime(path) > cache_mtime:
                return False

        if os.path.exists(csr_path) and os.path.getmtime(csr_path) > cache_mtime:
            return False
        if os.path.exists(csr_path_with_rank) and os.path.getmtime(csr_path_with_rank) > cache_mtime:
            return False

    return True

def save_raw_cases_to_cache(raw_cases: list, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(raw_cases, f, protocol=pickle.HIGHEST_PROTOCOL)
    log_print(f"[CACHE] データを {cache_path} にキャッシュ")

def load_raw_cases_from_cache(cache_path: str) -> list:
    with open(cache_path, "rb") as f:
        raw_cases = pickle.load(f)
    log_print(f"[CACHE] キャッシュ {cache_path} からデータを読み込み")
    return raw_cases

def compute_affine_fit(x_true_tensor, x_pred_tensor):
    xp = x_pred_tensor.detach().cpu().double().view(-1).numpy()
    yt = x_true_tensor.detach().cpu().double().view(-1).numpy()

    n = xp.size
    if n == 0:
        return 1.0, 0.0, float("nan"), float("nan")

    sx = xp.sum()
    sy = yt.sum()
    sxx = (xp * xp).sum()
    sxy = (xp * yt).sum()

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        a = 1.0
        b = 0.0
    else:
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n

    rmse_before = float(np.sqrt(((xp - yt) ** 2).mean()))
    rmse_after = float(np.sqrt(((a * xp + b - yt) ** 2).mean()))

    return a, b, rmse_before, rmse_after

def load_case_with_csr(gnn_dir: str, time_str: str, rank_str: str):
    p_path   = os.path.join(gnn_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
    x_path   = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")

    csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
    if not os.path.exists(csr_path):
        csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

    if not os.path.exists(p_path):
        raise FileNotFoundError(p_path)
    has_x_true = os.path.exists(x_path)
    if not os.path.exists(csr_path):
        raise FileNotFoundError(csr_path)

    with open(p_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    try:
        header_nc = lines[0].split()
        header_nf = lines[1].split()
        assert header_nc[0] == "nCells"
        assert header_nf[0] == "nFaces"
        nCells = int(header_nc[1])
    except Exception as e:
        raise RuntimeError(f"nCells/nFaces ヘッダの解釈に失敗: {p_path}\n{e}")

    try:
        idx_cells = next(i for i, ln in enumerate(lines) if ln.startswith("CELLS"))
        idx_edges = next(i for i, ln in enumerate(lines) if ln.startswith("EDGES"))
    except StopIteration:
        raise RuntimeError(f"CELLS/EDGES セクションが見つからない: {p_path}")

    idx_wall = None
    for i, ln in enumerate(lines):
        if ln.startswith("WALL_FACES"):
            idx_wall = i
            break
    if idx_wall is None:
        idx_wall = len(lines)

    cell_lines = lines[idx_cells + 1: idx_edges]
    edge_lines = lines[idx_edges + 1: idx_wall]

    if len(cell_lines) != nCells:
        log_print(f"[WARN] nCells: {nCells} と CELLS 行数: {len(cell_lines)} が違う (time={time_str}).")

    feats_np = np.zeros((len(cell_lines), 13), dtype=np.float32)
    b_np     = np.zeros(len(cell_lines), dtype=np.float32)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 14:
            raise RuntimeError(f"CELLS 行の列数が足りない: {ln}")
        cell_id = int(parts[0])
        xcoord  = float(parts[1])
        ycoord  = float(parts[2])
        zcoord  = float(parts[3])
        diag    = float(parts[4])
        b_val   = float(parts[5])
        skew    = float(parts[6])
        non_ortho  = float(parts[7])
        aspect     = float(parts[8])
        diag_con   = float(parts[9])
        V          = float(parts[10])
        h          = float(parts[11])
        size_jump  = float(parts[12])
        Co         = float(parts[13])

        if not (0 <= cell_id < len(cell_lines)):
            raise RuntimeError(f"cell_id の範囲がおかしい: {cell_id}")

        feats_np[cell_id, :] = np.array(
            [
                xcoord, ycoord, zcoord,
                diag, b_val, skew, non_ortho, aspect,
                diag_con, V, h, size_jump, Co
            ],
            dtype=np.float32
        )
        b_np[cell_id] = b_val

    e_src = []
    e_dst = []
    for ln in edge_lines:
        parts = ln.split()
        if parts[0] == "WALL_FACES":
            break
        if len(parts) != 5:
            raise RuntimeError(f"EDGES 行の列数が 5 ではない: {ln}")
        lower = int(parts[1])
        upper = int(parts[2])
        if not (0 <= lower < len(cell_lines) and 0 <= upper < len(cell_lines)):
            raise RuntimeError(f"lower/upper の cell index が範囲外: {ln}")

        e_src.append(lower)
        e_dst.append(upper)
        e_src.append(upper)
        e_dst.append(lower)

    edge_index_np = np.vstack([
        np.array(e_src, dtype=np.int64),
        np.array(e_dst, dtype=np.int64)
    ])

    if has_x_true:
        x_true_np = np.zeros(len(cell_lines), dtype=np.float32)
        with open(x_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) != 2:
                    raise RuntimeError(f"x_*.dat の行形式が想定外: {ln}")
                cid = int(parts[0])
                val = float(parts[1])
                if not (0 <= cid < len(cell_lines)):
                    raise RuntimeError(f"x_*.dat の cell id が範囲外: {cid}")
                x_true_np[cid] = val
    else:
        x_true_np = None

    with open(csr_path, "r") as f:
        csr_lines = [ln.strip() for ln in f if ln.strip()]

    try:
        h0 = csr_lines[0].split()
        h1 = csr_lines[1].split()
        h2 = csr_lines[2].split()
        assert h0[0] == "nRows"
        assert h1[0] == "nCols"
        assert h2[0] == "nnz"
        nRows = int(h0[1])
        nCols = int(h1[1])
        nnz   = int(h2[1])
    except Exception as e:
        raise RuntimeError(f"A_csr_{time_str}.dat のヘッダ解釈に失敗: {csr_path}\n{e}")

    if nRows != nCells:
        log_print(f"[WARN] CSR nRows={nRows} と pEqn nCells={nCells} が異なる (time={time_str}).")

    try:
        idx_rowptr = next(i for i, ln in enumerate(csr_lines) if ln.startswith("ROW_PTR"))
        idx_colind = next(i for i, ln in enumerate(csr_lines) if ln.startswith("COL_IND"))
        idx_vals   = next(i for i, ln in enumerate(csr_lines) if ln.startswith("VALUES"))
    except StopIteration:
        raise RuntimeError(f"ROW_PTR/COL_IND/VALUES が見つからない: {csr_path}")

    row_ptr_str = csr_lines[idx_rowptr + 1].split()
    col_ind_str = csr_lines[idx_colind + 1].split()
    vals_str    = csr_lines[idx_vals + 1].split()

    if len(row_ptr_str) != nRows + 1:
        raise RuntimeError(
            f"ROW_PTR の長さが nRows+1 と一致しない: len: {len(row_ptr_str)}, nRows: {nRows}"
        )
    if len(col_ind_str) != nnz:
        raise RuntimeError(
            f"COL_IND の長さが nnz と一致しない: len: {len(col_ind_str)}, nnz: {nnz}"
        )
    if len(vals_str) != nnz:
        raise RuntimeError(
            f"VALUES の長さが nnz と一致しない: len :{len(vals_str)}, nnz: {nnz}"
        )

    row_ptr_np = np.array(row_ptr_str, dtype=np.int64)
    col_ind_np = np.array(col_ind_str, dtype=np.int64)
    vals_np    = np.array(vals_str,    dtype=np.float32)

    row_idx_np = np.empty(nnz, dtype=np.int64)
    for i in range(nRows):
        start = row_ptr_np[i]
        end   = row_ptr_np[i+1]
        row_idx_np[start:end] = i

    return {
        "time": time_str,
        "rank": rank_str,
        "gnn_dir": gnn_dir,
        "feats_np": feats_np,
        "edge_index_np": edge_index_np,
        "x_true_np": x_true_np,
        "has_x_true": has_x_true,
        "b_np": b_np,
        "row_ptr_np": row_ptr_np,
        "col_ind_np": col_ind_np,
        "vals_np": vals_np,
        "row_idx_np": row_idx_np,
    }

class SimpleSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, 1))

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = F.relu(x)

        for i, (conv, norm) in enumerate(zip(self.convs[:-1], self.norms)):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = x + x_res

        x = self.convs[-1](x, edge_index)
        return x.view(-1)

def matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x):
    if x.dtype != vals.dtype:
        x = x.to(vals.dtype)

    n_rows = int(row_ptr.numel() - 1)
    y = torch.zeros(n_rows, device=x.device, dtype=x.dtype)
    y.index_add_(0, row_idx, vals * x[col_ind])
    return y

def matvec_csr_numpy(row_ptr, col_ind, vals, x):
    n = len(row_ptr) - 1
    A = csr_matrix((vals.astype(np.float64), col_ind, row_ptr), shape=(n, n))
    return A @ x.astype(np.float64)

def apply_diagonal_scaling_csr(row_ptr_np, col_ind_np, vals_np, diag_np, b_np):
    eps = 1e-12
    n = len(diag_np)

    diag_abs = np.abs(diag_np).astype(np.float64)
    diag_sqrt = np.sqrt(diag_abs + eps).astype(np.float32)
    diag_inv_sqrt = (1.0 / diag_sqrt).astype(np.float32)

    row_indices = np.repeat(np.arange(n, dtype=np.int64), np.diff(row_ptr_np))
    vals_scaled = (vals_np * diag_inv_sqrt[row_indices] * diag_inv_sqrt[col_ind_np]).astype(np.float32)

    b_scaled = b_np * diag_inv_sqrt

    return vals_scaled, b_scaled, diag_sqrt

def build_w_pde_from_feats(feats_np: np.ndarray,
                           w_pde_max: float = W_PDE_MAX,
                           use_mesh_quality_weights: bool = USE_MESH_QUALITY_WEIGHTS) -> np.ndarray:
    n_cells = feats_np.shape[0]

    if not use_mesh_quality_weights:
        return np.ones(n_cells, dtype=np.float32)

    skew      = feats_np[:, 5]
    non_ortho = feats_np[:, 6]
    aspect    = feats_np[:, 7]
    size_jump = feats_np[:, 11]

    SKEW_REF      = 0.2
    NONORTH_REF   = 10.0
    ASPECT_REF    = 5.0
    SIZEJUMP_REF  = 1.5

    q_skew      = np.clip(skew      / (SKEW_REF + 1e-12),     0.0, 5.0)
    q_non_ortho = np.clip(non_ortho / (NONORTH_REF + 1e-12),  0.0, 5.0)
    q_aspect    = np.clip(aspect    / (ASPECT_REF + 1e-12),   0.0, 5.0)
    q_sizeJump  = np.clip(size_jump / (SIZEJUMP_REF + 1e-12), 0.0, 5.0)

    w_raw = (
        1.0
        + 1.0 * (q_skew      - 1.0)
        + 1.0 * (q_non_ortho - 1.0)
        + 1.0 * (q_aspect    - 1.0)
        + 1.0 * (q_sizeJump  - 1.0)
    )

    w_clipped = np.clip(w_raw, 1.0, w_pde_max)

    return w_clipped.astype(np.float32)

def convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device, lazy_load=False,
                                    use_diagonal_scaling=USE_DIAGONAL_SCALING):
    feats_np  = rc["feats_np"]
    x_true_np = rc["x_true_np"]
    has_x_true = rc.get("has_x_true", x_true_np is not None)

    diag_np = feats_np[:, 3].copy()

    vals_np = rc["vals_np"]
    b_np = rc["b_np"]
    diag_sqrt_np = None

    if use_diagonal_scaling:
        vals_np, b_np, diag_sqrt_np = apply_diagonal_scaling_csr(
            rc["row_ptr_np"], rc["col_ind_np"], rc["vals_np"], diag_np, rc["b_np"]
        )

    feats_norm = (feats_np - feat_mean) / feat_std

    if has_x_true and x_true_np is not None:
        x_true_norm_np = (x_true_np - x_mean) / x_std
    else:
        x_true_norm_np = None

    w_pde_np = build_w_pde_from_feats(feats_np)

    target_device = torch.device("cpu") if lazy_load else device

    feats       = torch.from_numpy(feats_norm).float().to(target_device)
    edge_index  = torch.from_numpy(rc["edge_index_np"]).long().to(target_device)

    if has_x_true and x_true_np is not None:
        x_true      = torch.from_numpy(x_true_np).float().to(target_device)
        x_true_norm = torch.from_numpy(x_true_norm_np).float().to(target_device)
    else:
        x_true      = None
        x_true_norm = None

    b       = torch.from_numpy(b_np).float().to(target_device)
    row_ptr = torch.from_numpy(rc["row_ptr_np"]).long().to(target_device)
    col_ind = torch.from_numpy(rc["col_ind_np"]).long().to(target_device)
    vals    = torch.from_numpy(vals_np).float().to(target_device)
    row_idx = torch.from_numpy(rc["row_idx_np"]).long().to(target_device)

    w_pde = torch.from_numpy(w_pde_np).float().to(target_device)

    if diag_sqrt_np is not None:
        diag_sqrt = torch.from_numpy(diag_sqrt_np).float().to(target_device)
    else:
        diag_sqrt = None

    volume_np = feats_np[:, 9].copy()
    volume = torch.from_numpy(volume_np).float().to(target_device)

    diag = torch.from_numpy(diag_np).float().to(target_device)

    return {
        "time": rc["time"],
        "rank": rc["rank"],
        "gnn_dir": rc["gnn_dir"],
        "feats": feats,
        "edge_index": edge_index,
        "x_true": x_true,
        "x_true_norm": x_true_norm,
        "has_x_true": has_x_true,
        "b": b,
        "row_ptr": row_ptr,
        "col_ind": col_ind,
        "vals": vals,
        "row_idx": row_idx,
        "w_pde": w_pde,
        "w_pde_np": w_pde_np,
        "diag_sqrt": diag_sqrt,
        "diag_sqrt_np": diag_sqrt_np,
        "use_diagonal_scaling": use_diagonal_scaling,
        "volume": volume,
        "diag": diag,

        "coords_np": feats_np[:, 0:3].copy(),
        "skew_np": feats_np[:, 5].copy(),
        "non_ortho_np": feats_np[:, 6].copy(),
        "aspect_np": feats_np[:, 7].copy(),
        "size_jump_np": feats_np[:, 11].copy(),
    }

def move_case_to_device(cs, device):
    x_true = cs["x_true"]
    x_true_norm = cs["x_true_norm"]
    has_x_true = cs.get("has_x_true", x_true is not None)
    diag_sqrt = cs.get("diag_sqrt")

    return {
        "time": cs["time"],
        "rank": cs["rank"],
        "gnn_dir": cs["gnn_dir"],
        "feats": cs["feats"].to(device, non_blocking=True),
        "edge_index": cs["edge_index"].to(device, non_blocking=True),
        "x_true": x_true.to(device, non_blocking=True) if x_true is not None else None,
        "x_true_norm": x_true_norm.to(device, non_blocking=True) if x_true_norm is not None else None,
        "has_x_true": has_x_true,
        "b": cs["b"].to(device, non_blocking=True),
        "row_ptr": cs["row_ptr"].to(device, non_blocking=True),
        "col_ind": cs["col_ind"].to(device, non_blocking=True),
        "vals": cs["vals"].to(device, non_blocking=True),
        "row_idx": cs["row_idx"].to(device, non_blocking=True),
        "w_pde": cs["w_pde"].to(device, non_blocking=True),
        "w_pde_np": cs["w_pde_np"],
        "diag_sqrt": diag_sqrt.to(device, non_blocking=True) if diag_sqrt is not None else None,
        "diag_sqrt_np": cs.get("diag_sqrt_np"),
        "use_diagonal_scaling": cs.get("use_diagonal_scaling", False),
        "volume": cs["volume"].to(device, non_blocking=True),
        "diag": cs["diag"].to(device, non_blocking=True),
        "coords_np": cs["coords_np"],
        "skew_np": cs["skew_np"],
        "non_ortho_np": cs["non_ortho_np"],
        "aspect_np": cs["aspect_np"],
        "size_jump_np": cs["size_jump_np"],
    }

def evaluate_validation_cases(
    model,
    cases_val,
    device,
    x_std_t,
    x_mean_t,
    use_amp_actual,
):
    num_val = len(cases_val)
    if num_val == 0:
        return None, None, None, 0

    sum_rel_err_val = 0.0
    sum_R_pred_val = 0.0
    sum_rmse_val = 0.0
    num_val_with_x = 0

    with torch.no_grad():
        for cs in cases_val:
            if USE_LAZY_LOADING:
                cs_gpu = move_case_to_device(cs, device)
            else:
                cs_gpu = cs

            feats = cs_gpu["feats"]
            edge_index = cs_gpu["edge_index"]
            x_true = cs_gpu["x_true"]
            b = cs_gpu["b"]
            row_ptr = cs_gpu["row_ptr"]
            col_ind = cs_gpu["col_ind"]
            vals = cs_gpu["vals"]
            row_idx = cs_gpu["row_idx"]
            w_pde = cs_gpu["w_pde"]
            has_x_true = cs_gpu.get("has_x_true", x_true is not None)
            diag_sqrt = cs_gpu.get("diag_sqrt", None)
            use_dscale = cs_gpu.get("use_diagonal_scaling", False) and (diag_sqrt is not None)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp_actual):
                x_pred_norm = model(feats, edge_index)
                x_pred = x_pred_norm * x_std_t + x_mean_t

            if has_x_true and x_true is not None:
                x_pred_centered = x_pred - torch.mean(x_pred)
                x_true_centered = x_true - torch.mean(x_true)
                diff = x_pred_centered - x_true_centered
                rel_err = torch.norm(diff) / (torch.norm(x_true_centered) + EPS_DATA)
                N = x_true.shape[0]
                rmse = torch.sqrt(torch.sum(diff * diff) / N)
                sum_rel_err_val += rel_err.item()
                sum_rmse_val += rmse.item()
                num_val_with_x += 1

            x_for_pde = (x_pred * diag_sqrt) if use_dscale else x_pred
            Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_for_pde)
            r = Ax - b
            sqrt_w = torch.sqrt(w_pde)
            wr = sqrt_w * r
            wb = sqrt_w * b
            norm_wr = torch.norm(wr)
            norm_wb = torch.norm(wb) + EPS_RES
            R_pred = norm_wr / norm_wb

            sum_R_pred_val += R_pred.item()

            if USE_LAZY_LOADING:
                del cs_gpu

    avg_R_pred_val = sum_R_pred_val / num_val
    if num_val_with_x > 0:
        avg_rel_err_val = sum_rel_err_val / num_val_with_x
        avg_rmse_val = sum_rmse_val / num_val_with_x
    else:
        avg_rel_err_val = avg_R_pred_val
        avg_rmse_val = 0.0

    return avg_rel_err_val, avg_rmse_val, avg_R_pred_val, num_val_with_x

def write_vtk_polydata(filepath, coords, scalars_dict):
    n_points = coords.shape[0]

    with open(filepath, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Pressure field data\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {n_points} float\n")
        for i in range(n_points):
            f.write(f"{coords[i, 0]:.9e} {coords[i, 1]:.9e} {coords[i, 2]:.9e}\n")

        f.write(f"VERTICES {n_points} {n_points * 2}\n")
        for i in range(n_points):
            f.write(f"1 {i}\n")

        f.write(f"POINT_DATA {n_points}\n")
        for name, data in scalars_dict.items():
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in data:
                f.write(f"{val:.9e}\n")

def init_plot():
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle(
        f"データ損失係数: {LAMBDA_DATA:g}, PDE損失係数: {LAMBDA_PDE:g}",
        fontsize=12
    )

    return fig, axes

def update_plot(fig, axes, history):
    ax_loss, ax_rel = axes

    ax_loss.clear()
    ax_rel.clear()

    epochs = np.array(history["epoch"], dtype=np.int32)
    if len(epochs) == 0:
        return

    loss      = np.array(history["loss"], dtype=np.float64)
    data_loss = np.array(history["data_loss"], dtype=np.float64)
    pde_loss  = np.array(history["pde_loss"], dtype=np.float64)
    rel_tr    = np.array(history["rel_err_train"], dtype=np.float64)

    rel_val = np.array(
        [np.nan if v is None else float(v) for v in history["rel_err_val"]],
        dtype=np.float64
    )

    loss_safe      = np.clip(loss,      EPS_PLOT, None)
    data_loss_safe = np.clip(data_loss, EPS_PLOT, None)
    pde_loss_safe  = np.clip(pde_loss,  EPS_PLOT, None)
    rel_tr_safe    = np.clip(rel_tr,    EPS_PLOT, None)

    rel_val_safe = rel_val.copy()
    mask = np.isfinite(rel_val_safe)
    rel_val_safe[mask] = np.clip(rel_val_safe[mask], EPS_PLOT, None)

    ax_loss.plot(epochs, loss_safe,      label="総損失",      linewidth=2)
    ax_loss.plot(epochs, data_loss_safe, label="データ損失",  linewidth=1.5, linestyle="--")
    ax_loss.plot(epochs, pde_loss_safe,  label="PDE損失",    linewidth=1.5, linestyle="--")

    ax_loss.set_xlabel("エポック数")
    ax_loss.set_ylabel("損失")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_rel.plot(epochs, rel_tr_safe,  label="相対誤差（訓練データ）", linewidth=1.5)
    ax_rel.plot(epochs, rel_val_safe, label="相対誤差（テストデータ）", linewidth=1.5)

    ax_rel.set_xlabel("エポック数")
    ax_rel.set_ylabel("相対誤差")
    ax_rel.set_yscale("log")
    ax_rel.grid(True, alpha=0.3)
    ax_rel.legend()

    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])

    plt.pause(0.01)

def train_gnn_auto_trainval_pde_weighted(
    data_dir: str,
    *,
    enable_plot: bool = True,
    return_history: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global LOGGER_FILE

    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    lambda_data_tag = str(LAMBDA_DATA).replace('.', 'p')
    lambda_pde_tag  = str(LAMBDA_PDE).replace('.', 'p')

    log_filename = (
        f"log_"
        f"DATA{lambda_data_tag}_"
        f"PDE{lambda_pde_tag}.txt"
    )
    log_path = os.path.join(OUTPUT_DIR, log_filename)

    LOGGER_FILE = open(log_path, "w", buffering=1)

    start_time = time.time()

    log_print(f"[INFO] ログ: {log_path}")
    log_print(f"[INFO] デバイス : {device}")

    all_time_rank_tuples, missing_info = find_time_rank_list(data_dir)

    if not all_time_rank_tuples:
        if missing_info.get("no_gnn_dirs"):
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ ディレクトリが見つからない"
            )

        error_messages = []
        if missing_info.get("missing_csr"):
            error_messages.append("A_csr_*.dat が見つからない")
        if not missing_info.get("missing_csr"):
            error_messages.append("pEqn_*_rank*.dat が見つからない")

        if error_messages:
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ 内に " + " ".join(error_messages)
            )
        else:
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ 内にデータが見つからない"
            )

    if missing_info.get("missing_x"):
        num_missing_x = len(missing_info["missing_x"])
        log_print(f"[WARN] x_*_rank*.dat が {num_missing_x} データ見つからないので、教師なし学習モードで続行")

    all_ranks = sorted(set(r for _, r, _ in all_time_rank_tuples), key=int)
    all_times_unique = sorted(set(t for t, _, _ in all_time_rank_tuples), key=float)
    all_gnn_dirs = sorted(set(g for _, _, g in all_time_rank_tuples))
    log_print(f"[INFO] ランク数: {all_ranks}")
    log_print(f"[INFO] 時刻数: {all_times_unique[:10]}{'...' if len(all_times_unique) > 10 else ''}")
    log_print(f"[INFO] ディレクトリ数: {len(all_gnn_dirs)}")

    random.seed(RANDOM_SEED)
    random.shuffle(all_time_rank_tuples)

    all_time_rank_tuples = all_time_rank_tuples[:MAX_NUM_CASES]
    n_total = len(all_time_rank_tuples)
    n_train = max(1, int(n_total * TRAIN_FRACTION))
    n_val   = n_total - n_train

    tuples_train = all_time_rank_tuples[:n_train]
    tuples_val   = all_time_rank_tuples[n_train:]

    log_print(f"[INFO] (時刻, ランク番号) のペアの数: {n_total}")
    log_print(f"[INFO] 訓練データ: {n_train} データ, テストデータ: {n_val} データ")
    log_print("=== 訓練データ ===")
    for t, r, g in tuples_train:
        log_print(f"  時刻:{t}, ランク番号:{r}")
    log_print("=== テストデータ ===")
    if tuples_val:
        for t, r, g in tuples_val:
            log_print(f"  時刻:{t}, ランク番号:{r}")
    else:
        log_print("  (テストデータなし)")
    log_print("===========================================")

    raw_cases_all = []
    cache_path = _get_cache_path(data_dir, all_time_rank_tuples) if USE_DATA_CACHE else None

    if USE_DATA_CACHE and _is_cache_valid(cache_path, all_time_rank_tuples):
        raw_cases_all = load_raw_cases_from_cache(cache_path)
    else:
        for t, r, g in all_time_rank_tuples:
            log_print(f"[LOAD] 時刻:{t}, ランク番号:{r}")
            rc = load_case_with_csr(g, t, r)
            raw_cases_all.append(rc)

        if USE_DATA_CACHE:
            save_raw_cases_to_cache(raw_cases_all, cache_path)

    raw_cases_train = []
    raw_cases_val   = []
    train_set = set(tuples_train)

    for rc in raw_cases_all:
        key = (rc["time"], rc["rank"], rc["gnn_dir"])
        if key in train_set:
            raw_cases_train.append(rc)
        else:
            raw_cases_val.append(rc)

    nFeat = raw_cases_train[0]["feats_np"].shape[1]
    for rc in raw_cases_train + raw_cases_val:
        if rc["feats_np"].shape[1] != nFeat:
            raise RuntimeError("nFeatures が一致していない")

    total_cells = sum(rc["feats_np"].shape[0] for rc in raw_cases_train + raw_cases_val)
    log_print(f"[INFO] 特徴量数: {nFeat}, 総セル数: {total_cells}")

    cases_with_x = [rc for rc in (raw_cases_train + raw_cases_val) if rc.get("has_x_true", False)]
    unsupervised_mode = len(cases_with_x) == 0

    if unsupervised_mode:
        log_print("[INFO] *** 教師なし学習モード（PINNs）: x_*_rank*.dat が見つからない ***")
        log_print("[INFO] *** 損失関数は PDE 損失のみを使用 ***")

    all_feats = np.concatenate(
        [rc["feats_np"] for rc in (raw_cases_train + raw_cases_val)], axis=0
    )

    feat_mean = all_feats.mean(axis=0, keepdims=True)
    feat_std  = all_feats.std(axis=0, keepdims=True) + 1e-12

    if not unsupervised_mode:
        all_xtrue = np.concatenate(
            [rc["x_true_np"] for rc in cases_with_x], axis=0
        )
        x_mean = all_xtrue.mean()
        x_std  = all_xtrue.std() + 1e-12
        log_print(
            f"[INFO] x_true: "
            f"min:{all_xtrue.min():.3e}, max:{all_xtrue.max():.3e}, mean:{x_mean:.3e}"
        )
    else:
        all_b = np.concatenate([rc["b_np"] for rc in raw_cases_train], axis=0)
        all_diag = np.concatenate([rc["feats_np"][:, 3] for rc in raw_cases_train], axis=0)

        b_rms = np.sqrt(np.mean(all_b**2)) + 1e-12
        diag_rms = np.sqrt(np.mean(all_diag**2)) + 1e-12

        x_mean = 0.0
        x_std = b_rms / diag_rms
        log_print(
            f"[INFO] x_true 統計: 教師なし学習モード"
            f" mean={x_mean:.3e}, std={x_std:.3e} (b_rms={b_rms:.3e}, diag_rms={diag_rms:.3e})"
        )

    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t  = torch.tensor(x_std,  dtype=torch.float32, device=device)

    train_ranks = sorted({int(rc["rank"]) for rc in raw_cases_train})
    num_ranks = max(train_ranks) + 1

    sums   = np.zeros(num_ranks, dtype=np.float64)
    sqsums = np.zeros(num_ranks, dtype=np.float64)
    counts = np.zeros(num_ranks, dtype=np.int64)

    if not unsupervised_mode:
        for rc in raw_cases_train:
            if not rc.get("has_x_true", False):
                continue
            r = int(rc["rank"])
            x = rc["x_true_np"].astype(np.float64).reshape(-1)
            sums[r]   += x.sum()
            sqsums[r] += np.square(x).sum()
            counts[r] += x.size

    x_mean_rank = np.full(num_ranks, x_mean, dtype=np.float64)
    x_std_rank  = np.full(num_ranks, x_std,  dtype=np.float64)

    for r in range(num_ranks):
        if counts[r] > 0:
            mean_r = sums[r] / counts[r]
            var_r  = sqsums[r] / counts[r] - mean_r * mean_r
            std_r  = np.sqrt(max(var_r, 1e-24))
            x_mean_rank[r] = mean_r
            x_std_rank[r]  = std_r

    for r in range(num_ranks):
        log_print(
            f"  ランク番号:{r}: カウント:{counts[r]}, "
            f"平均:{x_mean_rank[r]:.3e}, 分散:{x_std_rank[r]:.3e}"
        )

    x_mean_rank_t = torch.from_numpy(x_mean_rank.astype(np.float32)).to(device)
    x_std_rank_t  = torch.from_numpy(x_std_rank.astype(np.float32)).to(device)

    cases_train = []
    cases_val   = []
    w_all_list  = []

    if USE_LAZY_LOADING:
        log_print("[INFO] 遅延 GPU 転送モード（データを CPU に保持し使用時のみ GPU へ転送）")

    for rc in raw_cases_train:
        cs = convert_raw_case_to_torch_case(
            rc, feat_mean, feat_std, x_mean, x_std, device,
            lazy_load=USE_LAZY_LOADING
        )
        cases_train.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    for rc in raw_cases_val:
        cs = convert_raw_case_to_torch_case(
            rc, feat_mean, feat_std, x_mean, x_std, device,
            lazy_load=USE_LAZY_LOADING
        )
        cases_val.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    if w_all_list:
        w_all = np.concatenate(w_all_list, axis=0)

        w_min  = float(w_all.min())
        w_max  = float(w_all.max())
        w_mean = float(w_all.mean())
        p50    = float(np.percentile(w_all, 50))
        p90    = float(np.percentile(w_all, 90))
        p99    = float(np.percentile(w_all, 99))

        log_print(f"  カウント : {w_all.size}")
        log_print(f"  最小値   : {w_min:.3e}")
        log_print(f"  平均値   : {w_mean:.3e}")
        log_print(f"  最大値   : {w_max:.3e}")
        log_print(f"  p50      : {p50:.3e}")
        log_print(f"  p90      : {p90:.3e}")
        log_print(f"  p99      : {p99:.3e}")
        log_print("==========================================================================")

    num_train = len(cases_train)
    num_val   = len(cases_val)
    num_train_with_x = sum(1 for cs in cases_train if cs.get("has_x_true", False))

    model = SimpleSAGE(
        in_channels=nFeat,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None
    scheduler_type = None
    if USE_ONE_CYCLE_LR:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=ONE_CYCLE_MAX_LR,
            total_steps=NUM_EPOCHS,
            pct_start=ONE_CYCLE_PCT_START,
            anneal_strategy='cos',
            div_factor=ONE_CYCLE_MAX_LR / LR,
            final_div_factor=1e4,
        )
        scheduler_type = "onecycle"
        log_print(f"[INFO] OneCycleLR スケジューラ: 有効")
    elif USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHED_FACTOR,
            patience=LR_SCHED_PATIENCE,
            min_lr=LR_SCHED_MIN_LR,
            verbose=False,
        )
        scheduler_type = "plateau"

    use_amp_actual = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp_actual)
    if use_amp_actual:
        log_print("[INFO] 混合精度学習 (AMP) : 有効")
    else:
        if USE_AMP and device.type != "cuda":
            log_print("[INFO] AMP は CUDA のみ有効（CPU では無効）")

    if LAMBDA_DATA == 0 and LAMBDA_PDE == 0:
        raise ValueError(
            "[ERROR] LAMBDA_DATA と LAMBDA_PDE が両方とも 0"
        )

    if LAMBDA_DATA == 0:
        learning_mode = "教師なし学習 (PDE 損失のみ)"
    elif LAMBDA_PDE == 0:
        learning_mode = "教師あり学習 (データ損失のみ)"
    else:
        learning_mode = "ハイブリッド学習"

    log_print(f"=== {learning_mode} ===")
    log_print(f"    データ損失の重み: {LAMBDA_DATA}, PDE 損失の重み: {LAMBDA_PDE}")

    fig, axes = (None, None)
    if enable_plot:
        fig, axes = init_plot()
    history = {
        "epoch": [],
        "loss": [],
        "data_loss": [],
        "pde_loss": [],
        "gauge_loss": [],
        "rel_err_train": [],
        "rel_err_val": [],
    }

    best_val_metric = float('inf')
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    if USE_EARLY_STOPPING:
        log_print(f"[INFO] アーリーストッピング: 有効 (patience: {EARLY_STOPPING_PATIENCE})")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if USE_LR_WARMUP and epoch <= LR_WARMUP_EPOCHS:
            warmup_factor = epoch / LR_WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * warmup_factor

        total_data_loss = 0.0
        total_pde_loss  = 0.0
        total_gauge_loss = 0.0
        sum_rel_err_tr  = 0.0
        sum_R_pred_tr   = 0.0
        sum_rmse_tr     = 0.0
        num_cases_with_x = 0

        for cs in cases_train:
            if USE_LAZY_LOADING:
                cs_gpu = move_case_to_device(cs, device)
            else:
                cs_gpu = cs

            feats       = cs_gpu["feats"]
            edge_index  = cs_gpu["edge_index"]
            x_true      = cs_gpu["x_true"]
            b           = cs_gpu["b"]
            row_ptr     = cs_gpu["row_ptr"]
            col_ind     = cs_gpu["col_ind"]
            vals        = cs_gpu["vals"]
            row_idx     = cs_gpu["row_idx"]
            w_pde       = cs_gpu["w_pde"]
            has_x_true  = cs_gpu.get("has_x_true", x_true is not None)
            diag_sqrt   = cs_gpu.get("diag_sqrt", None)
            use_dscale  = cs_gpu.get("use_diagonal_scaling", False) and (diag_sqrt is not None)
            volume      = cs_gpu["volume"]
            diag        = cs_gpu["diag"]

            with torch.amp.autocast(device_type='cuda', enabled=use_amp_actual):
                x_pred_norm = model(feats, edge_index)
                x_pred = x_pred_norm * x_std_t + x_mean_t

                if has_x_true and x_true is not None and LAMBDA_DATA > 0:
                    rank_id = int(cs["rank"])
                    mean_r  = x_mean_rank_t[rank_id]
                    std_r   = x_std_rank_t[rank_id]

                    x_true_norm_case = (x_true - mean_r) / (std_r + 1e-12)
                    x_pred_norm_case_for_loss = (x_pred - mean_r) / (std_r + 1e-12)

                    data_loss_case = F.mse_loss(
                        x_pred_norm_case_for_loss,
                        x_true_norm_case
                    )
                    num_cases_with_x += 1
                else:
                    data_loss_case = None

                if LAMBDA_PDE > 0:
                    x_for_pde = (x_pred * diag_sqrt) if use_dscale else x_pred
                    Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_for_pde)
                    r  = Ax - b

                    sqrt_w = torch.sqrt(w_pde)
                    wr = sqrt_w * r
                    wb = sqrt_w * b

                    if PDE_LOSS_NORMALIZATION == "relative":
                        norm_wr_sq = torch.sum(wr * wr)
                        norm_wb_sq = torch.sum(wb * wb) + EPS_RES
                        pde_loss_case = norm_wr_sq / norm_wb_sq
                    elif PDE_LOSS_NORMALIZATION == "row_diag":
                        diag_abs = torch.abs(diag) + EPS_RES
                        r_normalized = r / diag_abs
                        wr_norm = sqrt_w * r_normalized
                        pde_loss_case = torch.mean(wr_norm * wr_norm)
                    else:
                        wAx = sqrt_w * Ax
                        norm_wr = torch.norm(wr)
                        norm_scale = torch.sqrt(torch.norm(wAx)**2 + torch.norm(wb)**2) + EPS_RES
                        pde_loss_case = (norm_wr / norm_scale) ** 2

                    with torch.no_grad():
                        norm_wr_diag = torch.norm(sqrt_w * r)
                        norm_wb_diag = torch.norm(sqrt_w * b) + EPS_RES
                        R_pred = norm_wr_diag / norm_wb_diag
                else:
                    pde_loss_case = None
                    R_pred = torch.tensor(0.0, device=device)

                total_volume = torch.sum(volume) + EPS_RES
                weighted_mean = torch.sum(x_pred * volume) / total_volume
                gauge_loss_case = weighted_mean ** 2

            loss_case = torch.tensor(0.0, device=device, requires_grad=True)

            if pde_loss_case is not None:
                loss_case = loss_case + (LAMBDA_PDE / num_train) * pde_loss_case

            if LAMBDA_GAUGE > 0:
                loss_case = loss_case + (LAMBDA_GAUGE / num_train) * gauge_loss_case

            if data_loss_case is not None:
                loss_case = loss_case + (LAMBDA_DATA / num_train_with_x) * data_loss_case

            if loss_case.requires_grad:
                scaler.scale(loss_case).backward()

            if pde_loss_case is not None:
                total_pde_loss += float(pde_loss_case.detach().cpu())
            total_gauge_loss += float(gauge_loss_case.detach().cpu())
            if data_loss_case is not None:
                total_data_loss += float(data_loss_case.detach().cpu())

            with torch.no_grad():
                if has_x_true and x_true is not None:
                    x_pred_centered = x_pred - torch.mean(x_pred)
                    x_true_centered = x_true - torch.mean(x_true)
                    diff = x_pred_centered - x_true_centered
                    N = x_true.shape[0]
                    rel_err_case = torch.norm(diff) / (torch.norm(x_true_centered) + EPS_DATA)
                    rmse_case    = torch.sqrt(torch.sum(diff * diff) / N)
                    sum_rel_err_tr += rel_err_case.item()
                    sum_rmse_tr    += rmse_case.item()
                sum_R_pred_tr  += R_pred.detach().item()

            if USE_LAZY_LOADING:
                del cs_gpu

        if USE_GRAD_CLIP:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)

        scaler.step(optimizer)
        scaler.update()

        avg_pde_loss = total_pde_loss / max(1, num_train) if LAMBDA_PDE > 0 else 0.0
        avg_gauge_loss = total_gauge_loss / max(1, num_train) if LAMBDA_GAUGE > 0 else 0.0

        loss_value = 0.0
        if LAMBDA_PDE > 0:
            loss_value += LAMBDA_PDE * avg_pde_loss
        if LAMBDA_GAUGE > 0:
            loss_value += LAMBDA_GAUGE * avg_gauge_loss

        if unsupervised_mode or num_cases_with_x == 0 or LAMBDA_DATA == 0:
            avg_data_loss = 0.0
        else:
            avg_data_loss = total_data_loss / max(1, num_cases_with_x)
            loss_value += LAMBDA_DATA * avg_data_loss
        avg_rel_err_val = None
        avg_R_pred_val = None
        avg_rmse_val = None

        need_val_eval = num_val > 0 and (
            scheduler_type == "plateau" or
            USE_EARLY_STOPPING or
            epoch % PLOT_INTERVAL == 0
        )
        if need_val_eval:
            model.eval()
            avg_rel_err_val, avg_rmse_val, avg_R_pred_val, _ = evaluate_validation_cases(
                model, cases_val, device, x_std_t, x_mean_t, use_amp_actual
            )

        if scheduler is not None:
            if scheduler_type == "onecycle":
                scheduler.step()
            else:
                metric_for_scheduler = avg_rel_err_val if avg_rel_err_val is not None else float(loss_value)
                scheduler.step(metric_for_scheduler)

        if USE_EARLY_STOPPING:
            current_metric = avg_rel_err_val if avg_rel_err_val is not None else float(loss_value)

            if best_val_metric == float('inf'):
                is_improvement = True
            else:
                relative_improvement = (best_val_metric - current_metric) / (best_val_metric + 1e-12)
                is_improvement = relative_improvement > EARLY_STOPPING_MIN_DELTA

            if is_improvement:
                best_val_metric = current_metric
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                log_print(f"\n[EARLY STOPPING] {EARLY_STOPPING_PATIENCE} エポック改善なし。"
                         f"ベストエポック: {best_epoch} (metric: {best_val_metric:.4e})")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    log_print(f"[INFO] ベストモデル (エポック: {best_epoch}) を復元")
                break

        if epoch % PLOT_INTERVAL == 0 or epoch == 1:
            if unsupervised_mode or num_cases_with_x == 0:
                avg_rel_err_tr = sum_R_pred_tr / num_train
                avg_rmse_tr    = 0.0
            else:
                avg_rel_err_tr = sum_rel_err_tr / num_cases_with_x
                avg_rmse_tr    = sum_rmse_tr / num_cases_with_x
            avg_R_pred_tr  = sum_R_pred_tr / num_train

            current_lr = optimizer.param_groups[0]["lr"]

            if num_val > 0 and avg_rel_err_val is None:
                model.eval()
                avg_rel_err_val, avg_rmse_val, avg_R_pred_val, _ = evaluate_validation_cases(
                    model, cases_val, device, x_std_t, x_mean_t, use_amp_actual
                )

            history["epoch"].append(epoch)
            history["loss"].append(float(loss_value))
            history["data_loss"].append(float(LAMBDA_DATA * avg_data_loss))
            history["pde_loss"].append(float(LAMBDA_PDE * avg_pde_loss))
            history["gauge_loss"].append(float(LAMBDA_GAUGE * avg_gauge_loss))
            history["rel_err_train"].append(float(avg_rel_err_tr))
            history["rel_err_val"].append(None if avg_rel_err_val is None else float(avg_rel_err_val))

            if enable_plot:
                update_plot(fig, axes, history)

            log = (
                f"[エポック: {epoch:5d}] ロス: {loss_value:.4e}, "
                f"学習率: {current_lr:.3e}, "
                f"データ損失: {LAMBDA_DATA * avg_data_loss:.4e}, "
                f"PDE 損失: {LAMBDA_PDE * avg_pde_loss:.4e}, "
            )
            if unsupervised_mode or num_cases_with_x == 0:
                log += f"gauge_loss={LAMBDA_GAUGE * avg_gauge_loss:.4e}, "
            log += (
                f"訓練誤差（相対誤差、平均値）: {avg_rel_err_tr:.4e}, "
            )
            if avg_rel_err_val is not None:
                log += (
                    f" テスト誤差（相対誤差、平均値）: {avg_rel_err_val:.4e} "
                )
            log_print(log)

    if enable_plot and len(history["epoch"]) > 0:
        final_plot_filename = (
            f"training_history_"
            f"DATA{lambda_data_tag}_"
            f"PDE{lambda_pde_tag}.png"
        )
        final_plot_path = os.path.join(OUTPUT_DIR, final_plot_filename)

        update_plot(fig, axes, history)
        fig.savefig(final_plot_path, dpi=200, bbox_inches='tight')
        log_print(f"[INFO] Training history figure saved to {final_plot_path}")

    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60.0
    log_print(
        f"[INFO] 計算時間: {elapsed:.2f} s "
        f"(~{h:02d}:{m:02d}:{s:05.2f})"
    )

    if LOGGER_FILE is not None:
        LOGGER_FILE.close()
        LOGGER_FILE = None

    log_print("\n=== 検証結果 (訓練データ) ===")
    model.eval()

    for cs in cases_train:
        time_str   = cs["time"]
        rank_str   = cs["rank"]

        if USE_LAZY_LOADING:
            cs_gpu = move_case_to_device(cs, device)
        else:
            cs_gpu = cs

        feats      = cs_gpu["feats"]
        edge_index = cs_gpu["edge_index"]
        x_true     = cs_gpu["x_true"]
        b          = cs_gpu["b"]
        row_ptr    = cs_gpu["row_ptr"]
        col_ind    = cs_gpu["col_ind"]
        vals       = cs_gpu["vals"]
        row_idx    = cs_gpu["row_idx"]
        w_pde      = cs_gpu["w_pde"]
        has_x_true = cs_gpu.get("has_x_true", x_true is not None)
        diag_sqrt  = cs_gpu.get("diag_sqrt", None)
        use_dscale = cs_gpu.get("use_diagonal_scaling", False) and (diag_sqrt is not None)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=use_amp_actual):
                x_pred_norm = model(feats, edge_index)
                x_pred = x_pred_norm * x_std_t + x_mean_t

            x_for_pde = (x_pred * diag_sqrt) if use_dscale else x_pred
            Ax_pred_w = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_for_pde)
            r_pred_w  = Ax_pred_w - b
            sqrt_w    = torch.sqrt(w_pde)
            wr_pred   = sqrt_w * r_pred_w
            wb        = sqrt_w * b
            norm_wr   = torch.norm(wr_pred)
            norm_wb   = torch.norm(wb) + EPS_RES
            R_pred_w  = norm_wr / norm_wb

            Ax_pred = Ax_pred_w
            r_scaled = Ax_pred - b
            r_pred  = (diag_sqrt * r_scaled) if use_dscale else r_scaled
            norm_r_pred    = torch.norm(r_pred)
            max_abs_r_pred = torch.max(torch.abs(r_pred))
            b_phys         = (diag_sqrt * b) if use_dscale else b
            norm_b         = torch.norm(b_phys)
            norm_Ax_pred   = torch.norm((b_phys + r_pred))
            R_pred_over_b  = norm_r_pred / (norm_b + EPS_RES)
            R_pred_over_Ax = norm_r_pred / (norm_Ax_pred + EPS_RES)

            if has_x_true and x_true is not None:
                diff = x_pred - x_true
                N = x_true.shape[0]
                rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                rmse    = torch.sqrt(torch.sum(diff * diff) / N)

                x_true_for_pde = (x_true * diag_sqrt) if use_dscale else x_true
                Ax_true = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_true_for_pde)
                r_scaled_true = Ax_true - b
                r_true  = (diag_sqrt * r_scaled_true) if use_dscale else r_scaled_true
                norm_r_true    = torch.norm(r_true)
                max_abs_r_true = torch.max(torch.abs(r_true))
                norm_Ax_true   = torch.norm((b_phys + r_true))
                R_true_over_b  = norm_r_true / (norm_b + EPS_RES)
                R_true_over_Ax = norm_r_true / (norm_Ax_true + EPS_RES)

        if has_x_true and x_true is not None:
            log_print(
                f"時刻: {time_str}, ランク番号: {rank_str}): "
                f"相対誤差: {rel_err.item():.4e}, RMSE: {rmse.item():.4e}, "
                f"重み付き残差: {R_pred_w.item():.4e}"
            )
            log_print(f"    真値最小値: {x_true.min().item():.6e}, 真値最大値: {x_true.max().item():.6e}, "
                  f"平均: {x_true.mean().item():.6e}, ノルム: {torch.norm(x_true).item():.6e}")
            log_print(f"    予測値最小値: {x_pred.min().item():.6e}, 予測値最大値: {x_pred.max().item():.6e}, "
                  f"平均: {x_pred.mean().item():.6e}, ノルム={torch.norm(x_pred).item():.6e}")
            log_print(f"    正規化予測値最小値: {x_pred_norm.min().item():.6e}, "
                  f"正規化予測値最大値:{x_pred_norm.max().item():.6e}, 平均: {x_pred_norm.mean().item():.6e}")
            log_print(f"    差分 (予測値 - 真値): ノルム={torch.norm(diff).item():.6e}")
            log_print(f"    正規化パラメータの平均: {x_mean_t.item():.6e}, 標準偏差: {x_std_t.item():.6e}")

            log_print("    [PDE 残差の比較 (OpenFOAM との比較)]")
            log_print(
                "      GNN 予測: "
                f"||r||_2={norm_r_pred.item():.6e}, "
                f"max|r_i|={max_abs_r_pred.item():.6e}, "
                f"||r||/||b||={R_pred_over_b.item():.5f}, "
                f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
            )
            log_print(
                "      OpenFOAM: "
                f"||r||_2={norm_r_true.item():.6e}, "
                f"max|r_i|={max_abs_r_true.item():.6e}, "
                f"||r||/||b||={R_true_over_b.item():.5f}, "
                f"||r||/||Ax||={R_true_over_Ax.item():.5f}"
            )
        else:
            log_print(
                f"時刻: {time_str}, ランク番号: {rank_str}) [教師なし学習]: "
                f"重み付き残差: {R_pred_w.item():.4e}"
            )
            log_print(f"予測値最小値: {x_pred.min().item():.6e}, 予測値最大値: {x_pred.max().item():.6e}, "
                  f"平均: {x_pred.mean().item():.6e}, ノルム: {torch.norm(x_pred).item():.6e}")
            log_print(
                "    [PDE残差 (GNN)]"
                f" ||r||_2: {norm_r_pred.item():.6e}, "
                f"max|r_i|: {max_abs_r_pred.item():.6e}, "
                f"||r||/||b||: {R_pred_over_b.item():.5f}, "
                f"||r||/||Ax||: {R_pred_over_Ax.item():.5f}"
            )

        x_pred_np = x_pred.cpu().numpy().reshape(-1)
        out_path = os.path.join(OUTPUT_DIR, f"x_pred_train_{time_str}_rank{rank_str}.dat")
        with open(out_path, "w") as f:
            for i, val in enumerate(x_pred_np):
                f.write(f"{i} {val:.9e}\n")

        coords_np = cs["coords_np"]

        vtk_pred_path = os.path.join(OUTPUT_DIR, f"pressure_pred_train_{time_str}_rank{rank_str}.vtk")
        write_vtk_polydata(vtk_pred_path, coords_np, {"p_pred": x_pred_np})

        if has_x_true and x_true is not None:
            x_true_np = x_true.cpu().numpy().reshape(-1)

            vtk_true_path = os.path.join(OUTPUT_DIR, f"pressure_true_train_{time_str}_rank{rank_str}.vtk")
            write_vtk_polydata(vtk_true_path, coords_np, {"p_true": x_true_np})

            error_np = x_pred_np - x_true_np
            vtk_compare_path = os.path.join(OUTPUT_DIR, f"pressure_compare_train_{time_str}_rank{rank_str}.vtk")
            write_vtk_polydata(vtk_compare_path, coords_np, {
                "p_true": x_true_np,
                "p_pred": x_pred_np,
                "error": error_np
            })

        if USE_LAZY_LOADING:
            del cs_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if num_val > 0:
        log_print("\n=== 検証結果 (テストデータ) ===")

        for cs in cases_val:
            time_str   = cs["time"]
            rank_str   = cs["rank"]

            if USE_LAZY_LOADING:
                cs_gpu = move_case_to_device(cs, device)
            else:
                cs_gpu = cs

            feats      = cs_gpu["feats"]
            edge_index = cs_gpu["edge_index"]
            x_true     = cs_gpu["x_true"]
            b          = cs_gpu["b"]
            row_ptr    = cs_gpu["row_ptr"]
            col_ind    = cs_gpu["col_ind"]
            vals       = cs_gpu["vals"]
            row_idx    = cs_gpu["row_idx"]
            w_pde      = cs_gpu["w_pde"]
            has_x_true = cs_gpu.get("has_x_true", x_true is not None)
            diag_sqrt  = cs_gpu.get("diag_sqrt", None)
            use_dscale = cs_gpu.get("use_diagonal_scaling", False) and (diag_sqrt is not None)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=use_amp_actual):
                    x_pred_norm = model(feats, edge_index)
                    x_pred = x_pred_norm * x_std_t + x_mean_t

                x_for_pde = (x_pred * diag_sqrt) if use_dscale else x_pred
                Ax_pred_w = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_for_pde)
                r_pred_w  = Ax_pred_w - b
                sqrt_w    = torch.sqrt(w_pde)
                wr_pred   = sqrt_w * r_pred_w
                wb        = sqrt_w * b
                norm_wr   = torch.norm(wr_pred)
                norm_wb   = torch.norm(wb) + EPS_RES
                R_pred_w  = norm_wr / norm_wb

                Ax_pred = Ax_pred_w
                r_scaled = Ax_pred - b
                r_pred  = (diag_sqrt * r_scaled) if use_dscale else r_scaled
                norm_r_pred    = torch.norm(r_pred)
                max_abs_r_pred = torch.max(torch.abs(r_pred))
                b_phys         = (diag_sqrt * b) if use_dscale else b
                norm_b         = torch.norm(b_phys)
                norm_Ax_pred   = torch.norm((b_phys + r_pred))
                R_pred_over_b  = norm_r_pred / (norm_b + EPS_RES)
                R_pred_over_Ax = norm_r_pred / (norm_Ax_pred + EPS_RES)

                if has_x_true and x_true is not None:
                    diff = x_pred - x_true
                    N = x_true.shape[0]
                    rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                    rmse    = torch.sqrt(torch.sum(diff * diff) / N)

                    x_true_for_pde = (x_true * diag_sqrt) if use_dscale else x_true
                    Ax_true = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_true_for_pde)
                    r_scaled_true = Ax_true - b
                    r_true  = (diag_sqrt * r_scaled_true) if use_dscale else r_scaled_true
                    norm_r_true    = torch.norm(r_true)
                    max_abs_r_true = torch.max(torch.abs(r_true))
                    norm_Ax_true   = torch.norm((b_phys + r_true))
                    R_true_over_b  = norm_r_true / (norm_b + EPS_RES)
                    R_true_over_Ax = norm_r_true / (norm_Ax_true + EPS_RES)

            if has_x_true and x_true is not None:
                log_print(
                    f"時刻: {time_str}, ランク番号: {rank_str}): "
                    f"相対誤差: {rel_err.item():.4e}, RMSE: {rmse.item():.4e}, "
                    f"重み付き残差: {R_pred_w.item():.4e}"
                )
                log_print(f"真値最小値: {x_true.min().item():.6e}, 真値最大値: {x_true.max().item():.6e}, "
                      f"平均: {x_true.mean().item():.6e}, ノルム: {torch.norm(x_true).item():.6e}")
                log_print(f"予測値最小値: {x_pred.min().item():.6e}, 予測値最大値: {x_pred.max().item():.6e}, "
                      f"平均: {x_pred.mean().item():.6e}, ノルム: {torch.norm(x_pred).item():.6e}")
                log_print(f"正規化予測値最小値: {x_pred_norm.min().item():.6e}, "
                      f"正規化予測値最大値: {x_pred_norm.max().item():.6e}, 平均: {x_pred_norm.mean().item():.6e}")
                log_print(f"差分 (予測値 - 真値): ノルム: {torch.norm(diff).item():.6e}")
                log_print(f"正規化パラメータ: 平均: {x_mean_t.item():.6e}, 標準偏差: {x_std_t.item():.6e}")

                log_print("    [PDE 残差の比較 (OpenFOAM との比較)]")
                log_print(
                    "      GNN 予測: "
                    f"||r||_2: {norm_r_pred.item():.6e}, "
                    f"max|r_i|: {max_abs_r_pred.item():.6e}, "
                    f"||r||/||b||: {R_pred_over_b.item():.5f}, "
                    f"||r||/||Ax||: {R_pred_over_Ax.item():.5f}"
                )
                log_print(
                    "      OpenFOAM: "
                    f"||r||_2: {norm_r_true.item():.6e}, "
                    f"max|r_i|: {max_abs_r_true.item():.6e}, "
                    f"||r||/||b||: {R_true_over_b.item():.5f}, "
                    f"||r||/||Ax||: {R_true_over_Ax.item():.5f}"
                )
            else:
                log_print(
                    f"時刻: {time_str}, ランク番号: {rank_str}) [教師なし学習]: "
                    f"重み付き残差: {R_pred_w.item():.4e}"
                )
                log_print(f"予測値最小値: {x_pred.min().item():.6e}, 予測値最大値: {x_pred.max().item():.6e}, "
                      f"平均={x_pred.mean().item():.6e}, ノルム={torch.norm(x_pred).item():.6e}")
                log_print(
                    "    [PDE残差 (GNN)]"
                    f" ||r||_2: {norm_r_pred.item():.6e}, "
                    f"max|r_i|: {max_abs_r_pred.item():.6e}, "
                    f"||r||/||b||: {R_pred_over_b.item():.5f}, "
                    f"||r||/||Ax||: {R_pred_over_Ax.item():.5f}"
                )

            x_pred_np = x_pred.cpu().numpy().reshape(-1)
            out_path = os.path.join(OUTPUT_DIR, f"x_pred_val_{time_str}_rank{rank_str}.dat")
            with open(out_path, "w") as f:
                for i, val in enumerate(x_pred_np):
                    f.write(f"{i} {val:.9e}\n")

            coords_np = cs["coords_np"]

            vtk_pred_path = os.path.join(OUTPUT_DIR, f"pressure_pred_val_{time_str}_rank{rank_str}.vtk")
            write_vtk_polydata(vtk_pred_path, coords_np, {"p_pred": x_pred_np})

            if has_x_true and x_true is not None:
                x_true_np = x_true.cpu().numpy().reshape(-1)

                vtk_true_path = os.path.join(OUTPUT_DIR, f"pressure_true_val_{time_str}_rank{rank_str}.vtk")
                write_vtk_polydata(vtk_true_path, coords_np, {"p_true": x_true_np})

                error_np = x_pred_np - x_true_np
                vtk_compare_path = os.path.join(OUTPUT_DIR, f"pressure_compare_val_{time_str}_rank{rank_str}.vtk")
                write_vtk_polydata(vtk_compare_path, coords_np, {
                    "p_true": x_true_np,
                    "p_pred": x_pred_np,
                    "error": error_np
                })

            if USE_LAZY_LOADING:
                del cs_gpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if return_history:
        return history

if __name__ == "__main__":
    train_gnn_auto_trainval_pde_weighted(DATA_DIR)
