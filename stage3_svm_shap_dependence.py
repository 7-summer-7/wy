from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import shap
from scipy.interpolate import UnivariateSpline

from common_config import (
    SEED,
    FIG_DIR, DAT_DIR, TAB_DIR, log, set_seed
)

# Config: point to stage3 SVM
MODEL_PATH = os.path.join("results_article_style", "models", "stage3_best_features", "SVM.joblib")

TARGET_FEATURES = ["SALT", "25(OH)D3", "Age", "tIgE", "Duration"]

X_RANGE = {
    "SALT": (0, 100),
    "25(OH)D3": (0, 50),
    "Age": (0, 80),
    "tIgE": (0, 1800),
    "Duration": (0, 240),
}

# plotting knobs
GRID_N = 300
BOOTSTRAP_N = 400
CI = (2.5, 97.5)
BAND_SCALE = 4
SPLINE_S_FACTOR = 0.5

NEG_BG = "#dff0e6"
POS_BG = "#f7efd8"
BAND_COLOR = "#9aa0a6"
LINE_COLOR = "#2f66d0"
ZERO_LINE_COLOR = "#777777"
TP_TEXT_COLOR = "#c63b3b"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _prep_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[m], dtype=float)
    y = np.asarray(y[m], dtype=float)
    if x.size < 6:
        raise ValueError("Too few valid points to fit.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(np.diff(x) <= 0):
        eps = (np.nanmax(x) - np.nanmin(x) + 1.0) * 1e-12
        for i in range(1, len(x)):
            if x[i] <= x[i - 1]:
                x[i] = x[i - 1] + eps
    return x, y


def _fit_spline(x: np.ndarray, y: np.ndarray) -> UnivariateSpline:
    s = max(1e-12, SPLINE_S_FACTOR * len(x) * float(np.nanvar(y)))
    return UnivariateSpline(x, y, s=s, k=3)


def _bootstrap_band(x: np.ndarray, y: np.ndarray, xg: np.ndarray, rng: np.random.Generator):
    n = len(x)
    preds = np.full((BOOTSTRAP_N, len(xg)), np.nan, dtype=float)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        try:
            xb, yb = _prep_xy(xb, yb)
            spl = _fit_spline(xb, yb)
            preds[b] = spl(xg)
        except Exception:
            pass
    lo = np.nanpercentile(preds, CI[0], axis=0)
    hi = np.nanpercentile(preds, CI[1], axis=0)
    return lo, hi


def _zero_crossings(xg: np.ndarray, yg: np.ndarray) -> list[float]:
    xs = []
    for i in range(len(xg) - 1):
        y0, y1 = yg[i], yg[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0 or y1 == 0:
            continue
        if y0 * y1 < 0:
            x0, x1 = xg[i], xg[i + 1]
            xc = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
            xs.append(float(xc))
    return xs


def _pick_tipping_point_near_median(x: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> float | None:
    xs = _zero_crossings(xg, yg)
    if not xs:
        return None
    x_median = float(np.nanmedian(x))
    xs_arr = np.asarray(xs, dtype=float)
    return float(xs_arr[np.argmin(np.abs(xs_arr - x_median))])


def _shade_by_tipping_point(ax, tp: float, x_min: float, x_max: float) -> None:
    tp = float(np.clip(tp, x_min, x_max))
    ax.axvspan(x_min, tp, color=NEG_BG, alpha=1.0, zorder=0)
    ax.axvspan(tp, x_max, color=POS_BG, alpha=1.0, zorder=0)


def _label_by_tipping_point(ax, x: np.ndarray, tp: float, x_min: float, x_max: float) -> None:
    tp = float(np.clip(tp, x_min, x_max))
    n_neg = int(np.sum((x >= x_min) & (x < tp)))
    n_pos = int(np.sum((x >= tp) & (x <= x_max)))
    y_top = ax.get_ylim()[1]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_text = y_top - 0.15 * y_span
    x_neg_mid = (x_min + tp) / 2.0
    x_pos_mid = (tp + x_max) / 2.0
    if n_neg > 0:
        ax.text(x_neg_mid, y_text, f"Negative\n(n={n_neg})", ha="center", va="top", fontsize=9)
    if n_pos > 0:
        ax.text(x_pos_mid, y_text, f"Positive\n(n={n_pos})", ha="center", va="top", fontsize=9)


def plot_dependence_style(feat: str, x_raw: np.ndarray, shap_y: np.ndarray, out_path: str, xlim=None, ylim=None, title: str | None = None) -> None:
    rng = np.random.default_rng(SEED)
    x, y = _prep_xy(x_raw, shap_y)
    if xlim is None:
        x_min, x_max = float(np.nanquantile(x, 0.01)), float(np.nanquantile(x, 0.99))
    else:
        x_min, x_max = float(xlim[0]), float(xlim[1])
    xg = np.linspace(x_min, x_max, GRID_N)
    spl = _fit_spline(x, y)
    yg = spl(xg)
    lo, hi = _bootstrap_band(x, y, xg, rng)
    band_ok = np.isfinite(lo).any() and np.isfinite(hi).any()
    if band_ok and BAND_SCALE != 1.0:
        lo = yg - BAND_SCALE * (yg - lo)
        hi = yg + BAND_SCALE * (hi - yg)
    tp = _pick_tipping_point_near_median(x, xg, yg)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if tp is not None and np.isfinite(tp):
        _shade_by_tipping_point(ax, tp, x_min, x_max)
    else:
        ax.axvspan(x_min, x_max, color=POS_BG, alpha=1.0, zorder=0)
    if band_ok:
        ax.fill_between(xg, lo, hi, color=BAND_COLOR, alpha=0.30, linewidth=0, zorder=1)
    ax.plot(xg, yg, color=LINE_COLOR, linewidth=2.2, zorder=2)
    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=1.0, zorder=3)
    ax.set_xlim(x_min, x_max)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        y_min = np.nanmin(np.r_[yg, lo if band_ok else np.array([]), y])
        y_max = np.nanmax(np.r_[yg, hi if band_ok else np.array([]), y])
        pad = 0.08 * (y_max - y_min + 1e-12)
        ax.set_ylim(y_min - pad, y_max + pad)
    if tp is not None and np.isfinite(tp):
        _label_by_tipping_point(ax, x, tp, x_min, x_max)
    if tp is not None and np.isfinite(tp):
        y0, y1 = ax.get_ylim()
        y_span = y1 - y0
        text_y = y0 + 0.60 * y_span
        ax.annotate(
            "Tipping point",
            xy=(tp, 0),
            xytext=(tp + 0.06 * (x_max - x_min), text_y),
            color=TP_TEXT_COLOR,
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
            ha="left",
            va="center",
            zorder=5
        )
    ax.set_xlabel(feat)
    ax.set_ylabel("SHAP value")
    ax.set_title(title or f"SVM–SHAP dependence: {feat}")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close(fig)


def main():
    set_seed(SEED)
    _ensure_dir(FIG_DIR)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    log(f"Loaded model: {MODEL_PATH}")

    x_model_path = os.path.join(DAT_DIR, "X_train_final_std.csv")
    x_orig_path = os.path.join(DAT_DIR, "X_train_imp_orig.csv")

    if not os.path.exists(x_model_path):
        raise FileNotFoundError(f"Missing: {x_model_path}")
    if not os.path.exists(x_orig_path):
        raise FileNotFoundError(f"Missing: {x_orig_path}")

    X_model = pd.read_csv(x_model_path)
    X_orig = pd.read_csv(x_orig_path)

    # Align features: prefer model.feature_names_in_, else fall back to Stage2 RFE best features
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        miss = [c for c in feature_names if c not in X_model.columns]
        if miss:
            raise ValueError(f"X_model missing columns required by model: {miss}")
        X_model = X_model[feature_names]
        if all(c in X_orig.columns for c in feature_names):
            X_orig = X_orig[feature_names]
    else:
        # try to load shap_rfe_best_features.csv produced in stage2
        bf_path = os.path.join(TAB_DIR, "shap_rfe_best_features.csv")
        if os.path.exists(bf_path):
            best_features = pd.read_csv(bf_path)["feature"].tolist()
            feature_names = list(best_features)
            miss = [c for c in feature_names if c not in X_model.columns]
            if miss:
                raise ValueError(f"X_model missing columns required by best_features: {miss}")
            X_model = X_model[feature_names]
            if all(c in X_orig.columns for c in feature_names):
                X_orig = X_orig[feature_names]
        else:
            feature_names = list(X_model.columns)

    if len(X_model) != len(X_orig):
        raise ValueError(f"Row mismatch: X_model({len(X_model)}) vs X_orig({len(X_orig)})")

    # Kernel SHAP: use a small background sample for speed
    bg_n = min(100, len(X_model))
    X_bg = shap.sample(X_model, bg_n, random_state=SEED)

    def _f(z):
        return model.predict_proba(z)[:, 1]

    explainer = shap.KernelExplainer(_f, X_bg)
    log("Initialized KernelExplainer for SVM (may be slow).")

    # Compute SHAP values (nsamples controls speed/accuracy)
    nsamples = 200
    shap_vals = explainer.shap_values(X_model, nsamples=nsamples)

    # KernelExplainer often returns a 2D array for binary case
    shap_mat = np.asarray(shap_vals)
    if shap_mat.ndim == 3:
        # sometimes returns (classes, n_samples, n_features)
        shap_mat = shap_mat[-1]

    if shap_mat.ndim != 2 or shap_mat.shape[1] != len(feature_names):
        raise ValueError(f"Unexpected SHAP shape: {shap_mat.shape}, features={len(feature_names)}")

    for feat in TARGET_FEATURES:
        if feat not in feature_names:
            log(f"[WARN] Feature {feat} not found, skip.")
            continue

        idx = feature_names.index(feat)
        x_raw = X_orig[feat].values
        s_val = shap_mat[:, idx]

        xlim = X_RANGE.get(feat, None)
        ylim = (-0.1, 0.3) if feat == "Duration" else None

        out_path = os.path.join(FIG_DIR, f"stage3_svm_shap_dependence_{feat}.svg")

        plot_dependence_style(
            feat=feat,
            x_raw=x_raw,
            shap_y=s_val,
            out_path=out_path,
            xlim=xlim,
            ylim=ylim,
            title=None
        )

        log(f"Plotted SVM SHAP dependence for {feat} -> {out_path}")

    log("Stage3 SVM SHAP dependence: done.")


if __name__ == "__main__":
    main()
