# stage5_compare_shap.py
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # ✅ MUST: avoid Tkinter errors on Windows
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import roc_auc_score

from common_config import (
    SEED, FIG_DIR, TAB_DIR, MOD_DIR, DAT_DIR, SHAP_DIR,
    CONTINUOUS_COLS,
    GLOBAL_SHAP_EXPLAIN_N, GLOBAL_SHAP_BACKGROUND_N,
    KERNEL_SHAP_EXPLAIN_CAP, KERNEL_SHAP_NSAMPLES,
    load_processed, log, set_seed,
    load_predictions, save_predictions,
    bootstrap_metrics_ci, ci_to_row, save_csv,
    plot_roc_panel_with_ci, plot_calibration_panel_with_ci, plot_dca_panel_with_ci, plot_metrics_panel_with_ci,
    save_shap_cache, load_shap_cache, pick_explain_background,
    _extract_shap_array, shap_summary_svg, shap_force_waterfall_svg
)
import shap

BASE_MODELS = ["DT", "RF", "XGB", "SVM", "GBDT"]
COMPARE_ORDER = ["DT", "RF", "XGB", "SVM", "GBDT", "STACK"]


def _safe_expected_value(expected_value, class_idx: int = 1) -> float:
    """
    Robustly convert expected_value to scalar float.
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


def _ensure_preds(tag: str, model, Xtr, ytr, Xte, yte, Xex, yex):
    """Compute and save predictions only if not already cached."""
    try:
        load_predictions(tag, "test")
        return
    except Exception:
        pass

    p_tr = model.predict_proba(Xtr)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]
    p_ex = model.predict_proba(Xex)[:, 1] if Xex.shape[0] else np.array([], dtype=float)

    save_predictions(tag, "train", ytr, p_tr)
    save_predictions(tag, "test", yte, p_te)
    save_predictions(tag, "ext", yex, p_ex)


def main(force: bool = False):
    set_seed(SEED)

    bf_path = os.path.join(TAB_DIR, "shap_rfe_best_features.csv")
    if not os.path.exists(bf_path):
        raise FileNotFoundError("Missing best features. Run Stage2 first.")
    best_features = pd.read_csv(bf_path)["feature"].tolist()

    # load data
    X_train, X_test, X_ext, y_train, y_test, y_ext, fn = load_processed()
    Xtr = X_train[best_features].values
    Xte = X_test[best_features].values
    Xex = X_ext[best_features].values if X_ext.shape[0] else np.empty((0, len(best_features)))

    # load models
    base_dir = os.path.join(MOD_DIR, "stage3_best_features")
    stack_path = os.path.join(MOD_DIR, "stage4_stacking", "stacking.joblib")
    if not os.path.exists(stack_path):
        raise FileNotFoundError("Missing stacking model. Run Stage4 first.")

    base_models = {mn: joblib.load(os.path.join(base_dir, f"{mn}.joblib")) for mn in BASE_MODELS}
    stack = joblib.load(stack_path)

    # ensure predictions exist (stage5 tags)
    for mn in BASE_MODELS:
        _ensure_preds(f"stage5_{mn}", base_models[mn], Xtr, y_train, Xte, y_test, Xex, y_ext)
    _ensure_preds("stage5_STACK", stack, Xtr, y_train, Xte, y_test, Xex, y_ext)

    tags = [f"stage5_{m}" for m in BASE_MODELS] + ["stage5_STACK"]

    # ---- Compare panels (TEST & EXT) with CI
    log("Stage 5: plotting comparison panels (ROC/Calib/DCA) for TEST & EXT...")
    plot_roc_panel_with_ci(tags, "test", os.path.join(FIG_DIR, "roc_panel_test.svg"))
    plot_calibration_panel_with_ci(tags, "test", os.path.join(FIG_DIR, "calibration_panel_test.svg"))
    plot_dca_panel_with_ci(tags, "test", os.path.join(FIG_DIR, "dca_panel_test.svg"))

    plot_roc_panel_with_ci(tags, "ext", os.path.join(FIG_DIR, "roc_panel_ext.svg"))
    plot_calibration_panel_with_ci(tags, "ext", os.path.join(FIG_DIR, "calibration_panel_ext.svg"))
    plot_dca_panel_with_ci(tags, "ext", os.path.join(FIG_DIR, "dca_panel_ext.svg"))

    # ---- Compare metrics tables (TEST/EXT) + metrics panels
    def build_compare(split: str, out_csv: str):
        rows = []
        for mn in BASE_MODELS + ["STACK"]:
            tag = f"stage5_{mn}" if mn != "STACK" else "stage5_STACK"
            y, p = load_predictions(tag, split)
            if len(y) == 0:
                continue
            ci = bootstrap_metrics_ci(y, p)
            rows.append(ci_to_row(mn, ci))
        save_csv(pd.DataFrame(rows), out_csv)

    compare_test = os.path.join(TAB_DIR, "compare_test_metrics_ci.csv")
    compare_ext = os.path.join(TAB_DIR, "compare_ext_metrics_ci.csv")
    build_compare("test", compare_test)
    build_compare("ext", compare_ext)

    plot_metrics_panel_with_ci(
        compare_test,
        os.path.join(FIG_DIR, "metrics_panel_test.svg"),
        "Metrics comparison (95% CI) — TEST",
        order=COMPARE_ORDER
    )
    plot_metrics_panel_with_ci(
        compare_ext,
        os.path.join(FIG_DIR, "metrics_panel_ext.svg"),
        "Metrics comparison (95% CI) — EXTERNAL",
        order=COMPARE_ORDER
    )

    # =========================
    # SHAP - First layer base learners
    # =========================
    log("Stage 5: SHAP for base learners (cache + beeswarm + bar)...")
    X_train_df = X_train[best_features].copy()

    # Explain/background (train)
    ex_idx, bg_idx = pick_explain_background(
        len(X_train_df),
        explain_n=500,
        background_n=100,
        seed=SEED
    )
    X_explain = X_train_df.iloc[ex_idx]
    X_bg = X_train_df.iloc[bg_idx]

    for mn in BASE_MODELS:
        tag = f"stage5_base_{mn}_shap"
        npz = os.path.join(SHAP_DIR, f"{tag}.npz")

        if os.path.exists(npz) and not force:
            log(f"  {mn}: SHAP cached.")
            shap_vals, base_vals, X_mat, sidx, fns, extra = load_shap_cache(tag)
            shap_vals = _extract_shap_array(shap_vals, class_idx=1)  # <- crucial
            X_ex_df = pd.DataFrame(X_mat, columns=fns)
        else:
            log(f"  {mn}: computing SHAP ...")
            model = base_models[mn]

            if mn in ["DT", "RF", "XGB", "GBDT"]:
                explainer = shap.TreeExplainer(model)
                vals_obj = explainer.shap_values(X_explain)
                shap_vals = _extract_shap_array(vals_obj, class_idx=1)
                exp_val = _safe_expected_value(explainer.expected_value, class_idx=1)
                base_vals = np.full((shap_vals.shape[0],), exp_val, dtype=float)
                X_ex_df = X_explain.copy()

            else:
                # SVM kernel SHAP (cap)
                cap = min(KERNEL_SHAP_EXPLAIN_CAP, len(X_explain))
                X_ex_small = X_explain.iloc[:cap]

                f = lambda z: model.predict_proba(z)[:, 1]
                explainer = shap.KernelExplainer(f, X_bg.values)
                vals_obj = explainer.shap_values(X_ex_small.values, nsamples=KERNEL_SHAP_NSAMPLES)
                shap_vals = _extract_shap_array(vals_obj, class_idx=1)
                base_vals = np.full((shap_vals.shape[0],), float(explainer.expected_value), dtype=float)
                X_ex_df = X_ex_small.copy()

            save_shap_cache(
                tag,
                shap_values=shap_vals,
                base_values=base_vals,
                data_matrix=X_ex_df.values,
                feature_names=best_features,
                sample_indices=np.array(ex_idx[:len(X_ex_df)], dtype=int),
                extra={"model": mn}
            )

            shap_vals, base_vals, X_mat, sidx, fns, extra = load_shap_cache(tag)
            shap_vals = _extract_shap_array(shap_vals, class_idx=1)  # <- crucial
            X_ex_df = pd.DataFrame(X_mat, columns=fns)

        # mean importance table + plots
        mean_abs = np.mean(np.abs(shap_vals), axis=0).astype(float)  # must be 1D now
        mean_df = pd.DataFrame({"feature": fns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        save_csv(mean_df, os.path.join(TAB_DIR, f"shap_base_{mn}_mean.csv"))

        out_prefix = os.path.join(FIG_DIR, f"shap_base_{mn}")
        shap_summary_svg(shap_vals, X_ex_df, out_prefix, max_display=30)

    # =========================
    # SHAP - Meta learner (LogisticRegression) + force + waterfall
    # =========================
    log("Stage 5: SHAP for meta-learner (cached) + force/waterfall...")

    meta_tag = "stage5_meta_shap"
    meta_npz = os.path.join(SHAP_DIR, f"{meta_tag}.npz")

    meta_train = stack.transform(Xtr)  # often 12 cols = 6 estimators * 2 probs
    if meta_train.shape[1] == 12:
        meta_train_use = meta_train[:, 1::2]
        meta_names = [f"p_{n}" for n in BASE_MODELS]
    else:
        meta_train_use = meta_train
        meta_names = [f"meta_{i}" for i in range(meta_train_use.shape[1])]

    rng = np.random.default_rng(SEED)
    ex_meta = rng.choice(meta_train_use.shape[0], size=min(500, meta_train_use.shape[0]), replace=False)
    X_meta_ex = meta_train_use[ex_meta]

    if os.path.exists(meta_npz) and not force:
        shap_vals_m, base_vals_m, X_mat_m, sidx_m, fns_m, extra_m = load_shap_cache(meta_tag)
        shap_vals_m = _extract_shap_array(shap_vals_m, class_idx=1)
        X_meta_df = pd.DataFrame(X_mat_m, columns=fns_m)
    else:
        meta_lr = stack.final_estimator_
        explainer = shap.LinearExplainer(meta_lr, meta_train_use, feature_perturbation="interventional")
        vals_obj = explainer.shap_values(X_meta_ex)
        shap_vals_m = _extract_shap_array(vals_obj, class_idx=1)
        base_val = _safe_expected_value(explainer.expected_value, class_idx=1)
        base_vals_m = np.full((shap_vals_m.shape[0],), base_val, dtype=float)

        save_shap_cache(
            meta_tag,
            shap_values=shap_vals_m,
            base_values=base_vals_m,
            data_matrix=X_meta_ex,
            feature_names=meta_names,
            sample_indices=np.array(ex_meta, dtype=int),
            extra={"type": "meta", "model": "LogReg"}
        )
        shap_vals_m, base_vals_m, X_mat_m, sidx_m, fns_m, extra_m = load_shap_cache(meta_tag)
        shap_vals_m = _extract_shap_array(shap_vals_m, class_idx=1)
        X_meta_df = pd.DataFrame(X_mat_m, columns=fns_m)

    shap_summary_svg(shap_vals_m, X_meta_df, os.path.join(FIG_DIR, "shap_meta"), max_display=len(meta_names))

    mean_abs_m = np.mean(np.abs(shap_vals_m), axis=0).astype(float)
    save_csv(
        pd.DataFrame({"feature": meta_names, "mean_abs_shap": mean_abs_m}).sort_values("mean_abs_shap", ascending=False),
        os.path.join(TAB_DIR, "shap_meta_mean.csv")
    )

    # force/waterfall for test sample closest to 0.5 (meta level)
    y_stack_test, p_stack_test = load_predictions("stage5_STACK", "test")
    if len(p_stack_test):
        idx_mid = int(np.argmin(np.abs(p_stack_test - 0.5)))
        meta_test = stack.transform(Xte)
        meta_test_use = meta_test[:, 1::2] if meta_test.shape[1] == 12 else meta_test

        meta_lr = stack.final_estimator_
        explainer = shap.LinearExplainer(meta_lr, meta_train_use, feature_perturbation="interventional")
        vals_obj = explainer.shap_values(meta_test_use[idx_mid:idx_mid + 1])
        sv_one = _extract_shap_array(vals_obj, class_idx=1).reshape(-1)
        base_one = _safe_expected_value(explainer.expected_value, class_idx=1)
        x_one = meta_test_use[idx_mid]

        shap_force_waterfall_svg(
            sv_one, x_one, meta_names, base_one,
            os.path.join(FIG_DIR, "shap_meta_force_example.svg"),
            os.path.join(FIG_DIR, "shap_meta_waterfall_example.svg"),
        )

    # =========================
    # SHAP - Global stacking (KernelExplainer) explain=500 background=100
    # =========================
    log("Stage 5: GLOBAL stacking SHAP (KernelExplainer) explain=500 background=100 (cached)...")

    global_tag = "stage5_stacking_global_shap"
    global_npz = os.path.join(SHAP_DIR, f"{global_tag}.npz")

    ex_g, bg_g = pick_explain_background(
        len(X_train_df),
        explain_n=GLOBAL_SHAP_EXPLAIN_N,
        background_n=GLOBAL_SHAP_BACKGROUND_N,
        seed=SEED
    )
    X_ex_g = X_train_df.iloc[ex_g].values
    X_bg_g = X_train_df.iloc[bg_g].values

    if os.path.exists(global_npz) and not force:
        shap_vals_g, base_vals_g, X_mat_g, sidx_g, fns_g, extra_g = load_shap_cache(global_tag)
        shap_vals_g = _extract_shap_array(shap_vals_g, class_idx=1)
        X_g_df = pd.DataFrame(X_mat_g, columns=fns_g)
    else:
        f = lambda z: stack.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(f, X_bg_g)
        vals_obj = explainer.shap_values(X_ex_g, nsamples=KERNEL_SHAP_NSAMPLES)
        shap_vals_g = _extract_shap_array(vals_obj, class_idx=1)
        base_vals_g = np.full((shap_vals_g.shape[0],), float(explainer.expected_value), dtype=float)

        save_shap_cache(
            global_tag,
            shap_values=shap_vals_g,
            base_values=base_vals_g,
            data_matrix=X_ex_g,
            feature_names=best_features,
            sample_indices=np.array(ex_g, dtype=int),
            extra={"type": "global_stacking", "explain": GLOBAL_SHAP_EXPLAIN_N, "bg": GLOBAL_SHAP_BACKGROUND_N}
        )
        shap_vals_g, base_vals_g, X_mat_g, sidx_g, fns_g, extra_g = load_shap_cache(global_tag)
        shap_vals_g = _extract_shap_array(shap_vals_g, class_idx=1)
        X_g_df = pd.DataFrame(X_mat_g, columns=fns_g)

    shap_summary_svg(shap_vals_g, X_g_df, os.path.join(FIG_DIR, "shap_stacking_global"), max_display=30)

    mean_abs_g = np.mean(np.abs(shap_vals_g), axis=0).astype(float)
    save_csv(
        pd.DataFrame({"feature": best_features, "mean_abs_shap": mean_abs_g}).sort_values("mean_abs_shap", ascending=False),
        os.path.join(TAB_DIR, "shap_stacking_global_mean.csv")
    )

    # global force/waterfall for test sample near 0.5
    if len(p_stack_test):
        idx_mid = int(np.argmin(np.abs(p_stack_test - 0.5)))
        x_mid = Xte[idx_mid]
        try:
            f = lambda z: stack.predict_proba(z)[:, 1]
            explainer = shap.KernelExplainer(f, X_bg_g)
            vals_obj = explainer.shap_values(x_mid.reshape(1, -1), nsamples=KERNEL_SHAP_NSAMPLES)
            sv_one = _extract_shap_array(vals_obj, class_idx=1).reshape(-1)
            base_one = float(explainer.expected_value)

            shap_force_waterfall_svg(
                sv_one, x_mid, best_features, base_one,
                os.path.join(FIG_DIR, "shap_stacking_global_force_example.svg"),
                os.path.join(FIG_DIR, "shap_stacking_global_waterfall_example.svg"),
            )
        except Exception as e:
            log(f"WARNING: global force/waterfall failed: {e}")



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(force=args.force)

import matplotlib.pyplot as plt
plt.close("all")
