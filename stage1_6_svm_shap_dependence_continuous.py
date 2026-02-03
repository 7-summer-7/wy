# stage3_xgb_shap_dependence_scatter_only.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import shap

from common_config import (
    SEED,
    FIG_DIR, DAT_DIR, log, set_seed
)

# =========================
# Config
# =========================
MODEL_PATH = os.path.join(
    "results_article_style", "models", "stage1_all_features", "XGB.joblib"
)

TARGET_FEATURES = ["SALT", "25(OH)D3", "Age", "tIgE", "Duration"]

X_RANGE = {
    "SALT": (0, 100),
    "25(OH)D3": (0, 50),
    "Age": (0, 80),
    "tIgE": (0, 1800),
    "Duration": (0, 240),
}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main():
    set_seed(SEED)
    _ensure_dir(FIG_DIR)

    # -------------------------
    # Load model (stage3 XGB)
    # -------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    log(f"Loaded model: {MODEL_PATH}")

    # -------------------------
    # Load data used for model input + original-scale data for x-axis
    # -------------------------
    x_model_path = os.path.join(DAT_DIR, "X_train_final_std.csv")
    x_orig_path = os.path.join(DAT_DIR, "X_train_imp_orig.csv")

    if not os.path.exists(x_model_path):
        raise FileNotFoundError(f"Missing: {x_model_path} (model input features)")
    if not os.path.exists(x_orig_path):
        raise FileNotFoundError(f"Missing: {x_orig_path} (original-scale features)")

    X_model = pd.read_csv(x_model_path)
    X_orig = pd.read_csv(x_orig_path)

    # align columns to model expectation when available
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        missing = [c for c in feature_names if c not in X_model.columns]
        if missing:
            raise ValueError(f"X_model is missing columns required by model: {missing}")
        X_model = X_model[feature_names]
        if all(c in X_orig.columns for c in feature_names):
            X_orig = X_orig[feature_names]
    else:
        feature_names = list(X_model.columns)

    if len(X_model) != len(X_orig):
        raise ValueError(f"Row mismatch: X_model({len(X_model)}) vs X_orig({len(X_orig)})")

    # -------------------------
    # Compute SHAP values
    # -------------------------
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_model)

    # handle binary/multiclass outputs
    if isinstance(shap_vals, list):
        shap_vals_mat = np.asarray(shap_vals[1]) if len(shap_vals) >= 2 else np.asarray(shap_vals[0])
    else:
        shap_vals_mat = np.asarray(shap_vals)

    if shap_vals_mat.ndim != 2 or shap_vals_mat.shape[1] != len(feature_names):
        raise ValueError(
            f"Unexpected shap_vals shape: {shap_vals_mat.shape}, features: {len(feature_names)}"
        )

    # -------------------------
    # Plot SHAP dependence (scatter only)
    # -------------------------
    for feat in TARGET_FEATURES:
        if feat not in feature_names:
            log(f"[WARN] Feature {feat} not found in model features, skip.")
            continue

        idx = feature_names.index(feat)
        x_raw = X_orig[feat].values
        s_val = shap_vals_mat[:, idx]

        plt.figure(figsize=(5.5, 4.5))
        plt.scatter(x_raw, s_val, s=18, alpha=0.45, edgecolor="none")
        plt.axhline(0, color="black", linestyle="--", linewidth=1.0)

        if feat in X_RANGE:
            plt.xlim(*X_RANGE[feat])
        else:
            x_lo, x_hi = np.nanquantile(x_raw, 0.01), np.nanquantile(x_raw, 0.99)
            plt.xlim(x_lo, x_hi)

        if feat == "Duration":
            plt.ylim(-0.1, 0.25)  # 需要就改

        plt.xlabel(feat)
        plt.ylabel("SHAP value (impact on XGB output)")
        plt.title(f"Stage3 XGB–SHAP dependence: {feat}")

        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, f"stage3_xgb_shap_dependence_{feat}.svg")
        plt.savefig(out_path, format="svg")
        plt.close()

        log(f"Plotted SHAP dependence (scatter only) for {feat} -> {out_path}")

    log("Stage3 (scatter-only SHAP dependence): done.")


if __name__ == "__main__":
    main()
