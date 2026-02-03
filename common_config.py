# common_config.py
from __future__ import annotations

import os
import json
import time
import random
import warnings
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc as auc_fn,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

import shap

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
SEED = 3037
#3036枕部脱发最强；3037是毳毛最强
DATA_PATH = "data2.csv"
TARGET_COL = "revisit"
HOSP_COL = "hosp"
SITE_COL = "Site"

CONTINUOUS_COLS = [
    "Age","Duration","WBC","Baso","PLT","Eos","Hb","Neut",
    "FT4","FT3","TSH","C3","C4","IgM","IgG","IgA","tIgE","Ca","Ly","SALT","25(OH)D3","ESR"
]
BINARY_COLS = ["Sex","Mutiple","Yellow_dot","Black_dot","Excl_mark","Broken_hairs","Short_lanugo"]
CATEGORICAL_COLS = [SITE_COL]  # site is unordered 1/2/3/4

OUT_DIR = "results_article_style"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
MOD_DIR = os.path.join(OUT_DIR, "models")
LOG_DIR = os.path.join(OUT_DIR, "logs")
DAT_DIR = os.path.join(OUT_DIR, "data_exports")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
SHAP_DIR = os.path.join(OUT_DIR, "shap_cache")
CLUST_DIR = os.path.join(OUT_DIR, "clustering")

for d in [OUT_DIR, FIG_DIR, TAB_DIR, MOD_DIR, LOG_DIR, DAT_DIR, PRED_DIR, SHAP_DIR, CLUST_DIR]:
    os.makedirs(d, exist_ok=True)

# Stage 1
N_ITER = 100
CV_FOLDS = 5

# Bootstrap
N_BOOT = 1000

# Global stacking SHAP (confirmed)
GLOBAL_SHAP_EXPLAIN_N = 500
GLOBAL_SHAP_BACKGROUND_N = 100

# Kernel SHAP safeguards (SVM base)
KERNEL_SHAP_EXPLAIN_CAP = 300
KERNEL_SHAP_NSAMPLES = 200

# DCA thresholds
DCA_THRESHOLDS = np.linspace(0.1, 0.60, 90)

# K-prototypes
KPROTO_K_LIST = list(range(2, 9))  # 2..8
KPROTO_MAX_ITER = 50
KPROTO_N_INIT = 10

# =========================
# Hyperparameter spaces (from your figure)
# =========================
HP_SPACE = {
    "DT": {
        "criterion": ["entropy", "gini"],
        "max_depth": list(range(10, 301, 10)),
        "max_features": ["sqrt", "log2"],
        "max_leaf_nodes": list(range(10, 101, 10)),
        "min_samples_split": list(range(10, 101, 10)),
        "splitter": ["best", "random"],
        "min_samples_leaf": list(range(10, 101, 10)),
    },
    "RF": {
        "n_estimators": list(range(20, 101, 5)),
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(10, 101, 10)),
        "min_samples_leaf": list(range(10, 101, 10)),
        "min_samples_split": list(range(10, 101, 10)),
        "max_features": ["sqrt", "log2"],
    },
    "XGB": {
        "max_depth": list(range(20, 51, 5)),
        "min_child_weight": list(range(5, 21, 1)),
        "reg_lambda": list(range(10, 51, 5)),  # lambda in figure
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "learning_rate": [round(x, 2) for x in np.arange(0.01, 0.101, 0.01)],  # eta in figure
        "gamma": [0.001, 0.005, 0.01, 0.05, 0.1],
        "n_estimators": list(range(20, 51, 5)),
    },
    "LGBM": {
        "boosting_type": ["gbdt", "goss", "dart", "rf"],
        "num_leaves": list(range(5, 51, 5)),
        "max_depth": list(range(10, 51, 5)),
        "learning_rate": [round(x, 3) for x in np.arange(0.001, 0.051, 0.001)],
        "n_estimators": list(range(10, 51, 5)),
        "subsample": [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)],
        "colsample_bytree": [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)],
        "reg_alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
    },
    "SVM": {
        "C": [round(x, 2) for x in np.arange(0.01, 0.51, 0.01)],
        "gamma": [round(x, 2) for x in np.arange(0.01, 0.51, 0.01)],
        "kernel": ["linear", "rbf", "poly"],
        "degree": list(range(1, 11)),
    },
    "GBDT": {
        "n_estimators": list(range(10, 101, 10)),
        "learning_rate": [round(x, 2) for x in np.arange(0.01, 0.101, 0.01)],
        "max_depth": list(range(2, 21, 1)),
        "subsample": [0.4, 0.5, 0.6, 0.7, 0.8],
        "min_samples_split": list(range(2, 21, 2)),
        "min_samples_leaf": list(range(20, 51, 5)),
        "max_features": ["sqrt", "log2"],
    },
}


# =========================
# Basic IO + logging
# =========================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(os.path.join(LOG_DIR, "runtime_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    X_train = pd.read_csv(os.path.join(DAT_DIR, "X_train_final_std.csv"))
    X_test = pd.read_csv(os.path.join(DAT_DIR, "X_test_final_std.csv"))
    X_ext = pd.read_csv(os.path.join(DAT_DIR, "X_ext_final_std.csv"))
    y_train = np.load(os.path.join(DAT_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DAT_DIR, "y_test.npy"))
    y_ext = np.load(os.path.join(DAT_DIR, "y_ext.npy"))
    fn = load_json(os.path.join(DAT_DIR, "feature_names.json"))
    return X_train, X_test, X_ext, y_train, y_test, y_ext, fn


# =========================
# Model factory
# =========================
def get_model(model_name: str, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    if model_name == "DT":
        return DecisionTreeClassifier(random_state=SEED, **params)
    if model_name == "RF":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    if model_name == "XGB":
        base = dict(random_state=SEED, eval_metric="logloss", use_label_encoder=False, n_jobs=-1)
        base.update(params)
        return XGBClassifier(**base)
    if model_name == "LGBM":
        base = dict(random_state=SEED, n_jobs=-1, verbosity=-1)
        base.update(params)
        return LGBMClassifier(**base)
    if model_name == "SVM":
        base = dict(probability=True, random_state=SEED)
        base.update(params)
        return SVC(**base)
    if model_name == "GBDT":
        return GradientBoostingClassifier(random_state=SEED, **params)
    raise ValueError(f"Unknown model: {model_name}")


# =========================
# Predictions cache
# =========================
def save_predictions(tag: str, split: str, y_true: np.ndarray, y_proba: np.ndarray) -> None:
    os.makedirs(PRED_DIR, exist_ok=True)
    np.save(os.path.join(PRED_DIR, f"{tag}_{split}_y.npy"), y_true.astype(int))
    np.save(os.path.join(PRED_DIR, f"{tag}_{split}_proba.npy"), y_proba.astype(float))

def load_predictions(tag: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    y = np.load(os.path.join(PRED_DIR, f"{tag}_{split}_y.npy"))
    p = np.load(os.path.join(PRED_DIR, f"{tag}_{split}_proba.npy"))
    return y, p


# =========================
# Metrics + bootstrap CI
# =========================
def _spe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    denom = tn + fp
    return float(tn / denom) if denom > 0 else np.nan

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.6) -> Dict[str, float]:
    y_pred = (y_proba >= thr).astype(int)
    out = {
        "AUC": np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ACC": accuracy_score(y_true, y_pred),
        "PRE": precision_score(y_true, y_pred, zero_division=0),
        "SEN": recall_score(y_true, y_pred, zero_division=0),
        "SPE": _spe(y_true, y_pred),
    }
    try:
        out["AUC"] = roc_auc_score(y_true, y_proba)
    except Exception:
        out["AUC"] = np.nan
    return out

def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    thr: float = 0.5,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    keys = ["AUC", "F1", "ACC", "PRE", "SEN", "SPE"]
    samples: Dict[str, List[float]] = {k: [] for k in keys}
    auc_nan = 0
    spe_nan = 0

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_proba[idx]
        m = compute_metrics(yt, yp, thr=thr)
        for k in keys:
            v = m[k]
            if k == "AUC" and (v is None or (isinstance(v, float) and np.isnan(v))):
                auc_nan += 1
            if k == "SPE" and (v is None or (isinstance(v, float) and np.isnan(v))):
                spe_nan += 1
            samples[k].append(v)

    out: Dict[str, Any] = {}
    for k in keys:
        arr = np.array(samples[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[k] = {"median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        else:
            out[k] = {
                "median": float(np.median(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }

    out["_boot_meta"] = {"n_boot": n_boot, "seed": seed, "thr": thr, "auc_nan": auc_nan, "spe_nan": spe_nan}
    return out

def ci_to_row(model: str, ci: Dict[str, Any]) -> Dict[str, Any]:
    r = {"model": model}
    for met in ["AUC", "F1", "ACC", "PRE", "SEN", "SPE"]:
        r[f"{met}_median"] = ci[met]["median"]
        r[f"{met}_ci_low"] = ci[met]["ci_low"]
        r[f"{met}_ci_high"] = ci[met]["ci_high"]
    r["n_boot"] = ci["_boot_meta"]["n_boot"]
    r["thr"] = ci["_boot_meta"]["thr"]
    r["auc_nan"] = ci["_boot_meta"]["auc_nan"]
    r["spe_nan"] = ci["_boot_meta"]["spe_nan"]
    return r


# =========================
# Plotting with CI (ROC / Calib / DCA / Metrics panel)
# =========================
def _roc_band(y_true: np.ndarray, y_proba: np.ndarray, fpr_grid: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    tprs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        tpr_i = np.interp(fpr_grid, fpr, tpr)
        tpr_i[0] = 0.0
        tpr_i[-1] = 1.0
        tprs.append(tpr_i)
    if len(tprs) == 0:
        z = np.full_like(fpr_grid, np.nan, dtype=float)
        return z, z, z
    tprs = np.vstack(tprs)
    return (
        np.nanmedian(tprs, axis=0),
        np.nanpercentile(tprs, 2.5, axis=0),
        np.nanpercentile(tprs, 97.5, axis=0),
    )

def plot_roc_panel_with_ci(model_tags: List[str], split: str, out_svg: str, n_boot: int = N_BOOT, seed: int = SEED):
    fpr_grid = np.linspace(0, 1, 201)
    plt.figure(figsize=(9, 7))
    for tag in model_tags:
        y, p = load_predictions(tag, split)
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        auc_pt = roc_auc_score(y, p)
        t_med, t_lo, t_hi = _roc_band(y, p, fpr_grid, n_boot, seed)
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{tag} (AUC={auc_pt:.3f})")
        plt.fill_between(fpr_grid, t_lo, t_hi, alpha=0.12)
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (bootstrap CI) — {split}")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()

def _calib_band(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    # reference bin centers from full data
    frac_full, mean_full = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    centers = mean_full
    mats = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_proba[idx]
        frac, mean = calibration_curve(yt, yp, n_bins=n_bins, strategy="quantile")
        if len(mean) < 2:
            continue
        frac_i = np.interp(centers, mean, frac)
        mats.append(frac_i)
    if len(mats) == 0:
        z = np.full_like(centers, np.nan, dtype=float)
        return centers, z, z, z
    mats = np.vstack(mats)
    return (
        centers,
        np.nanmedian(mats, axis=0),
        np.nanpercentile(mats, 2.5, axis=0),
        np.nanpercentile(mats, 97.5, axis=0),
    )

def plot_calibration_panel_with_ci(model_tags: List[str], split: str, out_svg: str, n_bins: int = 10, n_boot: int = N_BOOT, seed: int = SEED):
    plt.figure(figsize=(9, 7))
    for tag in model_tags:
        y, p = load_predictions(tag, split)
        if len(y) == 0:
            continue
        frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        centers, med, lo, hi = _calib_band(y, p, n_bins, n_boot, seed)
        plt.plot(mean, frac, marker="o", linewidth=1.8, label=tag)
        plt.fill_between(centers, lo, hi, alpha=0.12)
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration (bootstrap CI) — {split}")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()

def dca_net_benefit(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(int)
    n = len(y_true)
    nb = np.zeros_like(thresholds, dtype=float)
    for i, pt in enumerate(thresholds):
        y_pred = (y_proba >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        nb[i] = (tp / n) - (fp / n) * (pt / (1 - pt))
    return nb

def _dca_band(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mats = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        mats.append(dca_net_benefit(y_true[idx], y_proba[idx], thresholds))
    mats = np.vstack(mats)
    return (
        np.nanmedian(mats, axis=0),
        np.nanpercentile(mats, 2.5, axis=0),
        np.nanpercentile(mats, 97.5, axis=0),
    )

def plot_dca_panel_with_ci(model_tags: List[str], split: str, out_svg: str, thresholds: np.ndarray = DCA_THRESHOLDS, n_boot: int = N_BOOT, seed: int = SEED):
    plt.figure(figsize=(9, 7))
    # baseline: treat all / treat none
    y0, _ = load_predictions(model_tags[0], split)
    if len(y0) == 0:
        return
    prev = y0.mean()
    treat_none = np.zeros_like(thresholds)
    treat_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, treat_none, linestyle="--", linewidth=1.2, label="Treat none")
    plt.plot(thresholds, treat_all, linestyle="--", linewidth=1.2, label="Treat all")

    for tag in model_tags:
        y, p = load_predictions(tag, split)
        if len(y) == 0:
            continue
        nb = dca_net_benefit(y, p, thresholds)
        med, lo, hi = _dca_band(y, p, thresholds, n_boot, seed)
        plt.plot(thresholds, nb, linewidth=1.8, label=tag)
        plt.fill_between(thresholds, lo, hi, alpha=0.12)

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"DCA (bootstrap CI) — {split}")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()

def plot_metrics_panel_with_ci(metrics_ci_csv: str, out_svg: str, title: str, order: List[str]):
    df = pd.read_csv(metrics_ci_csv).set_index("model").loc[order].reset_index()
    metrics = ["AUC","F1","ACC","PRE","SEN","SPE"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.1 * len(metrics)))
    for ax, met in zip(axes, metrics):
        med = df[f"{met}_median"].values
        lo = df[f"{met}_ci_low"].values
        hi = df[f"{met}_ci_high"].values
        y_pos = np.arange(len(df))
        ax.errorbar(med, y_pos, xerr=[med-lo, hi-med], fmt="o", capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["model"].values)
        ax.invert_yaxis()
        ax.set_xlabel(met)
        ax.grid(True, axis="x", alpha=0.2)
    fig.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(out_svg, format="svg")
    plt.close()


# =========================
# SHAP cache helpers
# =========================
def _extract_shap_array(shap_values_obj, class_idx: int = 1) -> np.ndarray:
    """
    Normalize SHAP outputs to 2D array: (n_samples, n_features).
    Compatible with:
    - list of arrays (old binary/multiclass)
    - ndarray 2D or 3D (newer shap may return (n, p, n_outputs))
    - shap.Explanation
    """
    # shap.Explanation
    if hasattr(shap_values_obj, "values"):
        vals = np.array(shap_values_obj.values)
    else:
        vals = shap_values_obj

    # list output: pick class
    if isinstance(vals, list):
        return np.array(vals[class_idx])

    vals = np.array(vals)

    # ndarray output
    if vals.ndim == 2:
        return vals

    if vals.ndim == 3:
        # assume last dim is outputs/classes
        if vals.shape[-1] <= class_idx:
            class_idx = vals.shape[-1] - 1
        return vals[:, :, class_idx]

    raise ValueError(f"Unsupported SHAP values shape: {vals.shape}")

def _safe_expected_value(expected_value, class_idx: int = 1) -> float:
    """
    Make expected_value a scalar float robustly.
    - scalar -> scalar
    - len==1 -> that one
    - len>=2 -> pick class_idx if possible else last
    """
    ev = expected_value
    if isinstance(ev, list):
        ev = np.array(ev, dtype=float)
    if isinstance(ev, np.ndarray):
        if ev.ndim == 0:
            return float(ev)
        if ev.size == 1:
            return float(ev.ravel()[0])
        if ev.size > class_idx:
            return float(ev.ravel()[class_idx])
        return float(ev.ravel()[-1])
    return float(ev)

def save_shap_cache(
    tag: str,
    shap_values: np.ndarray,
    base_values: np.ndarray,
    data_matrix: np.ndarray,
    feature_names: List[str],
    sample_indices: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(SHAP_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(SHAP_DIR, f"{tag}.npz"),
        shap_values=shap_values,
        base_values=base_values,
        data_matrix=data_matrix,
        sample_indices=sample_indices
    )
    save_json(
        {"feature_names": feature_names, "extra": extra or {}},
        os.path.join(SHAP_DIR, f"{tag}.json")
    )

def load_shap_cache(tag: str):
    z = np.load(os.path.join(SHAP_DIR, f"{tag}.npz"), allow_pickle=True)
    meta = load_json(os.path.join(SHAP_DIR, f"{tag}.json"))
    return (
        z["shap_values"],
        z["base_values"],
        z["data_matrix"],
        z["sample_indices"],
        meta["feature_names"],
        meta.get("extra", {})
    )

def pick_explain_background(n: int, explain_n: int, background_n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    explain_n = min(explain_n, n)
    background_n = min(background_n, n)
    ex = rng.choice(idx, size=explain_n, replace=False)
    remain = np.setdiff1d(idx, ex)
    if remain.size >= background_n:
        bg = rng.choice(remain, size=background_n, replace=False)
    else:
        bg = rng.choice(idx, size=background_n, replace=False)
    return ex, bg

def shap_summary_svg(shap_values: np.ndarray, X_df: pd.DataFrame, out_prefix: str, max_display: int = 30):
    plt.figure()
    shap.summary_plot(shap_values, X_df, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_prefix + "_beeswarm.svg", format="svg")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_prefix + "_mean_bar.svg", format="svg")
    plt.close()

def shap_force_waterfall_svg(
    shap_values_row: np.ndarray,
    X_row: np.ndarray,
    feature_names: List[str],
    base_value: float,
    out_force_svg: str,
    out_waterfall_svg: str,
):
    # Force plot (matplotlib backend)
    try:
        plt.figure()
        shap.force_plot(
            base_value, shap_values_row, X_row,
            feature_names=feature_names, matplotlib=True, show=False
        )
        plt.tight_layout()
        plt.savefig(out_force_svg, format="svg")
        plt.close()
    except Exception as e:
        log(f"WARNING: force plot failed: {e}")

    # Waterfall plot
    try:
        exp = shap.Explanation(
            values=shap_values_row,
            base_values=base_value,
            data=X_row,
            feature_names=feature_names,
        )
        plt.figure()
        shap.plots.waterfall(exp, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_waterfall_svg, format="svg")
        plt.close()
    except Exception as e:
        log(f"WARNING: waterfall plot failed: {e}")
