# stage4_stacking.py
from __future__ import annotations

import os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from common_config import (
    SEED, CV_FOLDS,
    TAB_DIR, MOD_DIR,
    load_processed, get_model, log, set_seed,
    save_csv, save_predictions,
    bootstrap_metrics_ci, ci_to_row
)

BASE_MODELS = ["DT","RF","XGB","SVM","GBDT"]

def main(force: bool = False):
    set_seed(SEED)

    out_test = os.path.join(TAB_DIR, "stage4_stacking_test_metrics_ci.csv")
    if os.path.exists(out_test) and not force:
        log("Stage 4: outputs exist; skip (use --force to redo).")
        return

    bf_path = os.path.join(TAB_DIR, "shap_rfe_best_features.csv")
    p_path = os.path.join(TAB_DIR, "stage1_all_features_best_params.csv")
    if not os.path.exists(bf_path) or not os.path.exists(p_path):
        raise FileNotFoundError("Need best_features + Stage1 best_params first.")

    best_features = pd.read_csv(bf_path)["feature"].tolist()
    df_params = pd.read_csv(p_path)
    params_map = {r["model"]: json.loads(r["best_params"]) for _, r in df_params.iterrows()}

    X_train, X_test, X_ext, y_train, y_test, y_ext, fn = load_processed()

    estimators = [(mn, get_model(mn, params_map[mn])) for mn in BASE_MODELS]
    final_est = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )

    stack.fit(X_train[best_features].values, y_train)

    p_tr = stack.predict_proba(X_train[best_features].values)[:, 1]
    p_te = stack.predict_proba(X_test[best_features].values)[:, 1]
    p_ex = stack.predict_proba(X_ext[best_features].values)[:, 1] if X_ext.shape[0] else np.array([], dtype=float)

    model_dir = os.path.join(MOD_DIR, "stage4_stacking")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(stack, os.path.join(model_dir, "stacking.joblib"))

    tag = "stage4_STACK"
    save_predictions(tag, "train", y_train, p_tr)
    save_predictions(tag, "test", y_test, p_te)
    save_predictions(tag, "ext", y_ext, p_ex)

    ci_tr = bootstrap_metrics_ci(y_train, p_tr)
    ci_te = bootstrap_metrics_ci(y_test, p_te)
    ci_ex = bootstrap_metrics_ci(y_ext, p_ex) if len(y_ext) else None

    save_csv(pd.DataFrame([ci_to_row("STACK", ci_tr)]), os.path.join(TAB_DIR, "stage4_stacking_train_metrics_ci.csv"))
    save_csv(pd.DataFrame([ci_to_row("STACK", ci_te)]), os.path.join(TAB_DIR, "stage4_stacking_test_metrics_ci.csv"))
    if ci_ex is not None:
        save_csv(pd.DataFrame([ci_to_row("STACK", ci_ex)]), os.path.join(TAB_DIR, "stage4_stacking_ext_metrics_ci.csv"))

    log("Stage 4: done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(force=args.force)
