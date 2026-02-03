# stage1_5_svm_global_shap_full_features.py
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from common_config import (
    SEED,
    TAB_DIR, FIG_DIR, SHAP_DIR, MOD_DIR,
    load_processed, log, set_seed,
    save_csv,
    save_shap_cache, load_shap_cache,
    pick_explain_background, _extract_shap_array
)

# =========================
# Config
# =========================
MODEL_NAME = "SVM"
EXPLAIN_N = 500        # 与论文级 SHAP 常用规模一致
BACKGROUND_N = 100
NSAMPLES = 200         # Kernel SHAP 采样
MAX_DISPLAY = 20       # 图中展示前 N 个特征


def main(force: bool = False):
    set_seed(SEED)

    shap_tag = "stage1_5_svm_full_features_global_shap"
    npz_path = os.path.join(SHAP_DIR, f"{shap_tag}.npz")

    # -------------------------
    # Load data
    # -------------------------
    X_train, X_test, X_ext, y_train, y_test, y_ext, fn = load_processed()
    feature_names = X_train.columns.tolist()

    # -------------------------
    # Load best SVM params (Stage1)
    # -------------------------
    params_path = os.path.join(TAB_DIR, "stage1_all_features_best_params.csv")
    if not os.path.exists(params_path):
        raise FileNotFoundError("Need stage1_all_features_best_params.csv first.")

    df_params = pd.read_csv(params_path)
    best_params = json.loads(
        df_params[df_params["model"] == MODEL_NAME].iloc[0]["best_params"]
    )

    # -------------------------
    # Load / fit SVM
    # -------------------------
    log("Stage1.5: fitting best SVM on FULL features.")
    model = joblib.load(
        os.path.join(MOD_DIR, "stage1_all_features", f"{MODEL_NAME}.joblib")
    )

    # -------------------------
    # SHAP
    # -------------------------
    if os.path.exists(npz_path) and not force:
        log("Stage1.5: loading cached SHAP.")
        shap_vals, base_vals, X_mat, sidx, fns, extra = load_shap_cache(shap_tag)
        X_explain_df = pd.DataFrame(X_mat, columns=fns)
    else:
        log("Stage1.5: computing Kernel SHAP (SVM, full features).")

        ex_idx, bg_idx = pick_explain_background(
            len(X_train),
            explain_n=min(EXPLAIN_N, len(X_train)),
            background_n=BACKGROUND_N,
            seed=SEED
        )

        X_ex = X_train.iloc[ex_idx]
        X_bg = X_train.iloc[bg_idx]

        f = lambda z: model.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(f, X_bg.values)

        shap_vals = explainer.shap_values(
            X_ex.values,
            nsamples=NSAMPLES
        )
        shap_vals = _extract_shap_array(shap_vals, class_idx=1)
        base_vals = np.full(
            (shap_vals.shape[0],),
            float(explainer.expected_value)
        )

        save_shap_cache(
            shap_tag,
            shap_values=shap_vals,
            base_values=base_vals,
            data_matrix=X_ex.values,
            feature_names=feature_names,
            sample_indices=ex_idx,
            extra={
                "model": MODEL_NAME,
                "stage": "stage1_5_full_features"
            }
        )

        shap_vals, base_vals, X_mat, sidx, fns, extra = load_shap_cache(shap_tag)
        X_explain_df = pd.DataFrame(X_mat, columns=fns)

    # -------------------------
    # SHAP importance table
    # -------------------------
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    imp_df = pd.DataFrame({
        "feature": fns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    out_tab = os.path.join(
        TAB_DIR,
        "stage1_5_svm_full_features_shap_importance.csv"
    )
    save_csv(imp_df, out_tab)

    # -------------------------
    # Plot: paper-style SHAP
    # -------------------------
    log("Stage1.5: plotting SHAP summary (paper-style).")

    plt.figure(figsize=(6.5, 8))
    shap.summary_plot(
        shap_vals,
        X_explain_df,
        feature_names=fns,
        max_display=MAX_DISPLAY,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "stage1_5_svm_full_features_shap_beeswarm.svg"),
        format="svg"
    )
    plt.close()

    log("Stage1.5: done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(force=args.force)
