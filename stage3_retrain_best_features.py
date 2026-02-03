# stage3_retrain_best_features.py
from __future__ import annotations

import os, json
import numpy as np
import pandas as pd
import joblib

from common_config import (
    SEED, TAB_DIR, MOD_DIR,
    load_processed, get_model, log, set_seed,
    save_csv, save_predictions,
    bootstrap_metrics_ci, ci_to_row
)

MODEL_LIST = ["DT","RF","XGB","SVM","GBDT"]

def main(force: bool = False):
    set_seed(SEED)

    out_test = os.path.join(TAB_DIR, "stage3_best_features_test_metrics_ci.csv")
    if os.path.exists(out_test) and not force:
        log("Stage 3: outputs exist; skip (use --force to redo).")
        return

    # prerequisites
    bf_path = os.path.join(TAB_DIR, "shap_rfe_best_features.csv")
    p_path = os.path.join(TAB_DIR, "stage1_all_features_best_params.csv")
    if not os.path.exists(bf_path) or not os.path.exists(p_path):
        raise FileNotFoundError("Need Stage2 best_features + Stage1 best_params first.")

    best_features = pd.read_csv(bf_path)["feature"].tolist()
    df_params = pd.read_csv(p_path)
    params_map = {r["model"]: json.loads(r["best_params"]) for _, r in df_params.iterrows()}

    X_train, X_test, X_ext, y_train, y_test, y_ext, fn = load_processed()

    model_dir = os.path.join(MOD_DIR, "stage3_best_features")
    os.makedirs(model_dir, exist_ok=True)

    rows_tr, rows_te, rows_ex = [], [], []

    for mn in MODEL_LIST:
        log(f"Stage 3: retrain {mn} with best features (no tuning)...")
        m = get_model(mn, params_map[mn])
        m.fit(X_train[best_features].values, y_train)

        joblib.dump(m, os.path.join(model_dir, f"{mn}.joblib"))

        p_tr = m.predict_proba(X_train[best_features].values)[:, 1]
        p_te = m.predict_proba(X_test[best_features].values)[:, 1]
        p_ex = m.predict_proba(X_ext[best_features].values)[:, 1] if X_ext.shape[0] else np.array([], dtype=float)

        tag = f"stage3_{mn}"
        save_predictions(tag, "train", y_train, p_tr)
        save_predictions(tag, "test", y_test, p_te)
        save_predictions(tag, "ext", y_ext, p_ex)

        ci_tr = bootstrap_metrics_ci(y_train, p_tr)
        ci_te = bootstrap_metrics_ci(y_test, p_te)
        ci_ex = bootstrap_metrics_ci(y_ext, p_ex) if len(y_ext) else None

        rows_tr.append(ci_to_row(mn, ci_tr))
        rows_te.append(ci_to_row(mn, ci_te))
        if ci_ex is not None:
            rows_ex.append(ci_to_row(mn, ci_ex))

    save_csv(pd.DataFrame(rows_tr), os.path.join(TAB_DIR, "stage3_best_features_train_metrics_ci.csv"))
    save_csv(pd.DataFrame(rows_te), os.path.join(TAB_DIR, "stage3_best_features_test_metrics_ci.csv"))
    if len(rows_ex):
        save_csv(pd.DataFrame(rows_ex), os.path.join(TAB_DIR, "stage3_best_features_ext_metrics_ci.csv"))

    log("Stage 3: done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(force=args.force)
