
from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

import shap

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from common_config import (
    SEED, CV_FOLDS,
    TAB_DIR, MOD_DIR, FIG_DIR, DAT_DIR, LOG_DIR,
    get_model, log, set_seed,
    save_json, save_csv,
    shap_force_waterfall_svg,
)

# =========================
# 1) Binary thresholds (paper-defined)
# =========================
THRESHOLDS = {
    "Age": 30,
    "tIgE": 200,
    "SALT": 25,
    "Duration": 24,
    "d5(OH)D3": 17,
}

# =========================
# 2) Fixed binary features (paper-defined)
# =========================
FIXED_BINARY_FEATURES = [
    "Mutiple",
    "Site_4.0",      # one-hot
    "Black_dot",
    "Excl_mark",
    "Short_lanugo"
]

BASE_MODELS = ["DT", "RF", "XGB", "SVM", "GBDT"]

# -------------------------
# Column name resolution
# -------------------------
# Map your paper names -> possible column names in exported CSVs
# (Exports are produced by stage0_preprocess.py and include final_feature_names.)
CONT_NAME_ALIASES = {
    "Age": ["age", "Age"],
    "tIgE": ["tIgE", "tige", "ige", "IGE", "Ige"],
    "SALT": ["SALT", "salt", "Salt"],
    "Duration": ["Duration", "dur_m", "duration", "dur"],
    "d5(OH)D3": ["d5(OH)D3", "25(OH)D3", "25(oh)d3", "d3", "D3"],
}

BIN_NAME_ALIASES = {
    "Mutiple": ["Mutiple", "mutiple", "multiple", "amt", "AMT"],
    "Black_dot": ["Black_dot", "black_dot", "BLACK_DOT"],
    "Excl_mark": ["Excl_mark", "excl_mark", "EXCL_MARK"],
    "Short_lanugo": ["Short_lanugo", "short_lanugo", "SHORT_LANUGO"],
    "Site_4.0": ["Site_4.0", "site_4.0", "site_4", "Site_4", "site_4"],
}

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return ""

def _ensure_cols(df: pd.DataFrame, need: dict[str, list[str]], kind: str) -> dict[str, str]:
    resolved = {}
    missing = []
    for k, aliases in need.items():
        col = _pick_col(df, aliases)
        if not col:
            missing.append(k)
        else:
            resolved[k] = col
    if missing:
        raise ValueError(f"Missing {kind} columns in exported data (could not resolve): {missing}")
    return resolved

def _binarize_continuous(df: pd.DataFrame, cont_map: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for k, thr in THRESHOLDS.items():
        col = cont_map[k]
        x = pd.to_numeric(df[col], errors="coerce")
        out[k] = (x >= float(thr)).astype(int)
    return out

def _select_fixed_binary(df: pd.DataFrame, bin_map: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for k in FIXED_BINARY_FEATURES:
        col = bin_map[k]
        out[k] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    # Ensure Site_4.0 is 0/1
    if "Occipital" in out.columns:
        out["Occipital"] = (out["Occipital"] > 0).astype(int)
    return out

def build_core_binary_matrix(df_imp_orig: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, str]]:
    cont_map = _ensure_cols(df_imp_orig, CONT_NAME_ALIASES, "continuous-to-binarize")
    bin_map  = _ensure_cols(df_imp_orig, BIN_NAME_ALIASES, "fixed-binary")

    Xb_cont = _binarize_continuous(df_imp_orig, cont_map)
    Xb_fix  = _select_fixed_binary(df_imp_orig, bin_map)

    # Final order: 5 thresholded + 5 fixed (exact as paper)
    feature_order = list(THRESHOLDS.keys()) + FIXED_BINARY_FEATURES
    X_final = pd.concat([Xb_cont, Xb_fix], axis=1)[feature_order].astype(int)
    return X_final, feature_order, cont_map, bin_map

def _load_stage0_exports() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    tr = pd.read_csv(os.path.join(DAT_DIR, "X_train_imp_orig.csv"))
    te = pd.read_csv(os.path.join(DAT_DIR, "X_test_imp_orig.csv"))
    ex_path = os.path.join(DAT_DIR, "X_ext_imp_orig.csv")
    ex = pd.read_csv(ex_path) if os.path.exists(ex_path) else pd.DataFrame(columns=tr.columns)

    ytr = np.load(os.path.join(DAT_DIR, "y_train.npy"))
    yte = np.load(os.path.join(DAT_DIR, "y_test.npy"))
    yex_path = os.path.join(DAT_DIR, "y_ext.npy")
    yex = np.load(yex_path) if os.path.exists(yex_path) else np.array([], dtype=int)
    return tr, te, ex, ytr, yte, yex

def _try_load_best_params() -> dict[str, dict]:
    """
    Optional: reuse previously tuned params (stage1_all_features_best_params.csv).
    Even though tuning was on full features, this gives stable defaults.
    If file not found, fall back to empty dicts.
    """
    path = os.path.join(TAB_DIR, "stage1_all_features_best_params.csv")
    if not os.path.exists(path):
        return {m: {} for m in BASE_MODELS}

    dfp = pd.read_csv(path)
    out = {m: {} for m in BASE_MODELS}
    for _, r in dfp.iterrows():
        mn = str(r["model"])
        if mn in out:
            try:
                out[mn] = json.loads(r["best_params"])
            except Exception:
                out[mn] = {}
    return out

def _pick_single_sample(X: pd.DataFrame, model, strategy: str = "closest_to_0.5") -> int:
    p = model.predict_proba(X.values)[:, 1]
    if strategy == "max":
        return int(np.argmax(p))
    if strategy == "min":
        return int(np.argmin(p))
    # default: closest to 0.5
    return int(np.argmin(np.abs(p - 0.5)))

def main(force: bool = False, sample_split: str = "test", sample_strategy: str = "closest_to_0.5"):
    set_seed(SEED)

    model_dir = os.path.join(MOD_DIR, "binary_core_models")
    os.makedirs(model_dir, exist_ok=True)

    cfg_path = os.path.join(model_dir, "binary_core_config.json")
    stack_path = os.path.join(model_dir, "stacking.joblib")

    if (os.path.exists(cfg_path) and os.path.exists(stack_path)) and not force:
        log("Binary-core retrain: outputs exist; skip (use --force to redo).")
        return

    # ---- load exports from stage0
    df_tr, df_te, df_ex, ytr, yte, yex = _load_stage0_exports()

    # ---- build 10-feature binary matrices
    Xtr, feature_order, cont_map, bin_map = build_core_binary_matrix(df_tr)
    Xte, _, _, _ = build_core_binary_matrix(df_te)
    Xex = pd.DataFrame(columns=feature_order)
    if len(df_ex):
        Xex, _, _, _ = build_core_binary_matrix(df_ex)

    # Save binary matrices (optional, helpful for traceability)
    out_bin_dir = os.path.join(DAT_DIR, "binary_core")
    os.makedirs(out_bin_dir, exist_ok=True)
    Xtr.to_csv(os.path.join(out_bin_dir, "X_train_binary_core.csv"), index=False)
    Xte.to_csv(os.path.join(out_bin_dir, "X_test_binary_core.csv"), index=False)
    Xex.to_csv(os.path.join(out_bin_dir, "X_ext_binary_core.csv"), index=False)

    # ---- params (optional reuse)
    params_map = _try_load_best_params()

    # ---- train base models
    base_models = {}
    metrics_rows = []

    for mn in BASE_MODELS:
        log(f"Binary-core retrain: fitting {mn} ...")
        m = get_model(mn, params_map.get(mn, {}))
        m.fit(Xtr.values, ytr)
        joblib.dump(m, os.path.join(model_dir, f"{mn}.joblib"))
        base_models[mn] = m

        p_tr = m.predict_proba(Xtr.values)[:, 1]
        p_te = m.predict_proba(Xte.values)[:, 1]
        auc_tr = float(roc_auc_score(ytr, p_tr)) if len(np.unique(ytr)) > 1 else np.nan
        auc_te = float(roc_auc_score(yte, p_te)) if len(np.unique(yte)) > 1 else np.nan

        row = {"model": mn, "AUC_train": auc_tr, "AUC_test": auc_te}
        if len(yex):
            p_ex = m.predict_proba(Xex.values)[:, 1]
            row["AUC_ext"] = float(roc_auc_score(yex, p_ex)) if len(np.unique(yex)) > 1 else np.nan
        metrics_rows.append(row)

    save_csv(pd.DataFrame(metrics_rows), os.path.join(TAB_DIR, "binary_core_base_models_auc.csv"))

    # ---- stacking (LR meta)
    log("Binary-core retrain: fitting STACK (LR meta, 5-fold CV stacking) ...")
    estimators = [(mn, base_models[mn]) for mn in BASE_MODELS]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    final_est = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )
    stack.fit(Xtr.values, ytr)
    joblib.dump(stack, stack_path)

    # ---- persist config for web app
    config = {
        "feature_order": feature_order,
        "thresholds": THRESHOLDS,
        "fixed_binary_features": FIXED_BINARY_FEATURES,
        "resolved_continuous_columns": cont_map,   # in exported CSVs
        "resolved_fixed_binary_columns": bin_map,  # in exported CSVs
        "model_dir": model_dir,
        "stacking_model_path": stack_path,
        "note": "Input to web calculator should be 10 binary values in feature_order."
    }
    save_json(config, cfg_path)

    # ---- (optional) SVM single-sample SHAP force + waterfall on selected sample
    # Kernel SHAP is used for SVM. We compute only 1 sample explanation to keep runtime acceptable.
    try:
        log("Binary-core retrain: generating SVM single-sample SHAP force/waterfall (SVG) ...")
        svm = base_models["SVM"]

        if sample_split.lower() == "train":
            X_ref, y_ref = Xtr, ytr
        elif sample_split.lower() == "ext" and len(yex):
            X_ref, y_ref = Xex, yex
        else:
            X_ref, y_ref = Xte, yte

        idx_one = _pick_single_sample(X_ref, svm, strategy=sample_strategy)
        x_one = X_ref.iloc[idx_one].values.astype(float)

        # background: up to 100 rows sampled from train
        rng = np.random.default_rng(SEED)
        bg_n = min(100, len(Xtr))
        bg_idx = rng.choice(len(Xtr), size=bg_n, replace=False) if len(Xtr) > bg_n else np.arange(len(Xtr))
        X_bg = Xtr.iloc[bg_idx].values.astype(float)

        f = lambda z: svm.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(f, X_bg)
        vals_obj = explainer.shap_values(x_one.reshape(1, -1), nsamples=200)

        # Normalize to 1D
        if hasattr(vals_obj, "values"):
            sv = np.array(vals_obj.values)
        else:
            sv = np.array(vals_obj)
        if isinstance(sv, list):
            sv = np.array(sv[0])
        sv = sv.reshape(-1)

        base_val = float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) else float(np.array(explainer.expected_value).ravel()[-1])

        out_force = os.path.join(FIG_DIR, "svm_binary_core_force_example.svg")
        out_wf = os.path.join(FIG_DIR, "svm_binary_core_waterfall_example.svg")
        shap_force_waterfall_svg(
            shap_values_row=sv,
            X_row=x_one,
            feature_names=feature_order,
            base_value=base_val,
            out_force_svg=out_force,
            out_waterfall_svg=out_wf,
        )
        log(f"  Saved: {out_force}")
        log(f"  Saved: {out_wf}")
    except Exception as e:
        log(f"WARNING: SVM SHAP force/waterfall generation failed: {e}")

    log("Binary-core retrain: done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="retrain and overwrite existing outputs")
    ap.add_argument("--sample", type=str, default="test", choices=["train", "test", "ext"], help="which split to pick the SHAP example from")
    ap.add_argument("--strategy", type=str, default="closest_to_0.5", choices=["closest_to_0.5", "max", "min"], help="how to pick the example sample")
    args = ap.parse_args()

    main(force=args.force, sample_split=args.sample, sample_strategy=args.strategy)
