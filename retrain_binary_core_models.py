# retrain_binary_core_models.py
# -*- coding: utf-8 -*-
"""
Binary-core (10 features) retraining + evaluation + stacking.

What this script does
---------------------
1) Load stage0 exports (original-scale, imputed):
     DAT_DIR/X_train_imp_orig.csv, X_test_imp_orig.csv, X_ext_imp_orig.csv
     DAT_DIR/y_train.npy, y_test.npy, y_ext.npy
2) Convert 5 key continuous variables into binary variables using fixed thresholds.
3) Combine with 5 fixed binary variables (including Site_4.0 one-hot).
4) Retrain five base models (DT/RF/XGB/SVM/GBDT) and a STACK model (LR meta learner; 5-fold CV stacking).
5) Save models + config for the web calculator.
6) Save predictions (train/test/ext) into PRED_DIR with tags "bincore_<MODEL>".
7) Generate comparison performance table (TEST, and EXT if available) for six metrics:
     AUC, F1, ACC, PRE, SEN, SPE
   with bootstrap 95% CI (same method as your pipeline).

Usage
-----
python retrain_binary_core_models.py --force
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from common_config import (
    SEED, CV_FOLDS, N_BOOT,
    TAB_DIR, MOD_DIR, DAT_DIR,
    get_model, log, set_seed,
    save_json, save_csv,
    save_predictions,
    bootstrap_metrics_ci, ci_to_row,
)

# =========================
# 1) Binary thresholds (paper-defined)
# =========================
THRESHOLDS = {
    "Age": 30,
    "tIgE": 200,
    "SALT": 25,
    "Duration": 18,
    "25(OH)D3": 17,
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
COMPARE_ORDER = ["DT", "RF", "XGB", "SVM", "GBDT", "STACK"]

# -------------------------
# Column name resolution
# -------------------------
CONT_NAME_ALIASES = {
    "Age": ["age", "Age"],
    "tIgE": ["tIgE", "tige", "ige", "IGE", "Ige"],
    "SALT": ["SALT", "salt", "Salt"],
    "Duration": ["Duration", "dur_m", "duration", "dur"],
    "25(OH)D3": ["d5(OH)D3", "25(OH)D3", "25(oh)d3", "d3", "D3"],
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
    if "Site_4.0" in out.columns:
        out["Site_4.0"] = (out["Site_4.0"] > 0).astype(int)
    return out

def build_core_binary_matrix(df_imp_orig: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, str]]:
    cont_map = _ensure_cols(df_imp_orig, CONT_NAME_ALIASES, "continuous-to-binarize")
    bin_map  = _ensure_cols(df_imp_orig, BIN_NAME_ALIASES, "fixed-binary")

    Xb_cont = _binarize_continuous(df_imp_orig, cont_map)
    Xb_fix  = _select_fixed_binary(df_imp_orig, bin_map)

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
    Optional: reuse stage1 tuned params (from full-features tuning). If not present, use defaults.
    """
    path = os.path.join(TAB_DIR, "stage1_all_features_best_params.csv")
    if not os.path.exists(path):
        return {m: {} for m in BASE_MODELS}

    dfp = pd.read_csv(path)
    out = {m: {} for m in BASE_MODELS}
    for _, r in dfp.iterrows():
        mn = str(r.get("model"))
        if mn in out:
            try:
                out[mn] = json.loads(r.get("best_params"))
            except Exception:
                out[mn] = {}
    return out

def _build_metrics_table_from_preds(split: str, tag_map: dict[str, str], out_csv: str) -> pd.DataFrame:
    """
    tag_map: display_name -> prediction_tag_prefix (used by save_predictions in this script)
    """
    rows = []
    for name in COMPARE_ORDER:
        tag = tag_map.get(name)
        if not tag:
            continue
        # predictions are saved as <tag>_<split>_y.npy / proba.npy in PRED_DIR via common_config.save_predictions
        from common_config import load_predictions
        y, p = load_predictions(tag, split)
        if len(y) == 0:
            continue
        ci = bootstrap_metrics_ci(y, p, n_boot=N_BOOT, seed=SEED, thr=0.5)
        rows.append(ci_to_row(name, ci))
    df = pd.DataFrame(rows)
    save_csv(df, out_csv)
    return df

def main(force: bool = False):
    set_seed(SEED)

    model_dir = os.path.join(MOD_DIR, "binary_core_models")
    os.makedirs(model_dir, exist_ok=True)

    cfg_path = os.path.join(model_dir, "binary_core_config.json")
    stack_path = os.path.join(model_dir, "stacking.joblib")

    out_compare_test = os.path.join(TAB_DIR, "binary_core_compare_test_metrics_ci.csv")
    out_compare_ext  = os.path.join(TAB_DIR, "binary_core_compare_ext_metrics_ci.csv")

    if (os.path.exists(cfg_path) and os.path.exists(stack_path) and os.path.exists(out_compare_test)) and not force:
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

    # Save binary matrices (for reproducibility / web background)
    out_bin_dir = os.path.join(DAT_DIR, "binary_core")
    os.makedirs(out_bin_dir, exist_ok=True)
    Xtr.to_csv(os.path.join(out_bin_dir, "X_train_binary_core.csv"), index=False)
    Xte.to_csv(os.path.join(out_bin_dir, "X_test_binary_core.csv"), index=False)
    Xex.to_csv(os.path.join(out_bin_dir, "X_ext_binary_core.csv"), index=False)

    # ---- params (optional reuse)
    params_map = _try_load_best_params()

    # ---- train base models + save predictions
    base_models = {}
    tag_map = {}  # display name -> tag
    for mn in BASE_MODELS:
        log(f"Binary-core retrain: fitting {mn} ...")
        m = get_model(mn, params_map.get(mn, {}))
        m.fit(Xtr.values, ytr)
        joblib.dump(m, os.path.join(model_dir, f"{mn}.joblib"))
        base_models[mn] = m

        tag = f"bincore_{mn}"
        tag_map[mn] = tag

        p_tr = m.predict_proba(Xtr.values)[:, 1]
        p_te = m.predict_proba(Xte.values)[:, 1]
        p_ex = m.predict_proba(Xex.values)[:, 1] if len(yex) else np.array([], dtype=float)

        save_predictions(tag, "train", ytr, p_tr)
        save_predictions(tag, "test", yte, p_te)
        save_predictions(tag, "ext", yex, p_ex)

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

    tag_stack = "bincore_STACK"
    tag_map["STACK"] = tag_stack

    p_tr = stack.predict_proba(Xtr.values)[:, 1]
    p_te = stack.predict_proba(Xte.values)[:, 1]
    p_ex = stack.predict_proba(Xex.values)[:, 1] if len(yex) else np.array([], dtype=float)

    save_predictions(tag_stack, "train", ytr, p_tr)
    save_predictions(tag_stack, "test", yte, p_te)
    save_predictions(tag_stack, "ext", yex, p_ex)

    # ---- persist config for web app
    config = {
        "feature_order": feature_order,
        "thresholds": THRESHOLDS,
        "fixed_binary_features": FIXED_BINARY_FEATURES,
        "resolved_continuous_columns": cont_map,   # in exported CSVs
        "resolved_fixed_binary_columns": bin_map,  # in exported CSVs
        "model_dir": model_dir,
        "stacking_model_path": stack_path,
        "svm_model_path": os.path.join(model_dir, "SVM.joblib"),
        "train_binary_core_csv": os.path.join(out_bin_dir, "X_train_binary_core.csv"),
        "note": "Web calculator inputs are 10 binary values in feature_order; SVM SHAP is computed at binary-core feature level."
    }
    save_json(config, cfg_path)

    # ---- comparison performance tables (bootstrap CI)
    log("Binary-core retrain: building TEST metrics comparison table (bootstrap 95% CI) ...")
    _build_metrics_table_from_preds("test", tag_map, out_compare_test)

    if len(yex):
        log("Binary-core retrain: building EXT metrics comparison table (bootstrap 95% CI) ...")
        _build_metrics_table_from_preds("ext", tag_map, out_compare_ext)

    log("Binary-core retrain: done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="retrain and overwrite existing outputs")
    args = ap.parse_args()
    main(force=args.force)
