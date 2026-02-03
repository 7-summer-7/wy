# stage1_tune_full.py
from __future__ import annotations

import os
import json
import time
import numpy as np
import pandas as pd

from common_config import (
    SEED, N_ITER, CV_FOLDS, N_BOOT,
    TAB_DIR, MOD_DIR, FIG_DIR,
    HP_SPACE, get_model, load_processed, log, set_seed,
    save_csv, save_predictions,
    bootstrap_metrics_ci, ci_to_row,
    plot_roc_panel_with_ci, plot_calibration_panel_with_ci, plot_dca_panel_with_ci,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

MODEL_LIST = ["DT", "RF", "XGB", "SVM", "GBDT"]

def main(force: bool = False):
    set_seed(SEED)

    out_params = os.path.join(TAB_DIR, "stage1_all_features_best_params.csv")
    if os.path.exists(out_params) and not force:
        log("Stage 1: outputs exist; skip (use --force to redo).")
        return

    X_train, X_test, X_ext, y_train, y_test, y_ext, fn = load_processed()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    best_rows = []
    train_rows = []
    test_rows = []
    ext_rows = []

    model_dir = os.path.join(MOD_DIR, "stage1_all_features")
    os.makedirs(model_dir, exist_ok=True)

    for mn in MODEL_LIST:
        log(f"Stage 1: tuning {mn} ...")
        t0 = time.time()

        model = get_model(mn)
        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions=HP_SPACE[mn],
            n_iter=N_ITER,
            scoring="roc_auc",
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            refit=True
        )
        rs.fit(X_train.values, y_train)

        best_est = rs.best_estimator_
        best_params = rs.best_params_
        best_cv_auc = float(rs.best_score_)
        runtime = time.time() - t0

        # Save model
        import joblib
        joblib.dump(best_est, os.path.join(model_dir, f"{mn}.joblib"))

        # Predictions
        p_tr = best_est.predict_proba(X_train.values)[:, 1]
        p_te = best_est.predict_proba(X_test.values)[:, 1]
        p_ex = best_est.predict_proba(X_ext.values)[:, 1] if X_ext.shape[0] else np.array([], dtype=float)

        tag = f"stage1_{mn}"
        save_predictions(tag, "train", y_train, p_tr)
        save_predictions(tag, "test", y_test, p_te)
        save_predictions(tag, "ext", y_ext, p_ex)

        # Bootstrap CI
        ci_tr = bootstrap_metrics_ci(y_train, p_tr, n_boot=N_BOOT, seed=SEED)
        ci_te = bootstrap_metrics_ci(y_test, p_te, n_boot=N_BOOT, seed=SEED)
        ci_ex = bootstrap_metrics_ci(y_ext, p_ex, n_boot=N_BOOT, seed=SEED) if len(y_ext) else None

        best_rows.append({
            "model": mn,
            "best_params": json.dumps(best_params, ensure_ascii=False),
            "cv_best_mean_auc": best_cv_auc,
            "runtime_sec": runtime
        })
        train_rows.append(ci_to_row(mn, ci_tr))
        test_rows.append(ci_to_row(mn, ci_te))
        if ci_ex is not None:
            ext_rows.append(ci_to_row(mn, ci_ex))

        log(f"Stage 1: {mn} done. CV AUC={best_cv_auc:.4f} runtime={runtime:.1f}s")

    save_csv(pd.DataFrame(best_rows), out_params)
    save_csv(pd.DataFrame(train_rows), os.path.join(TAB_DIR, "stage1_all_features_train_metrics_ci.csv"))
    save_csv(pd.DataFrame(test_rows), os.path.join(TAB_DIR, "stage1_all_features_test_metrics_ci.csv"))
    if len(ext_rows):
        save_csv(pd.DataFrame(ext_rows), os.path.join(TAB_DIR, "stage1_all_features_ext_metrics_ci.csv"))

    # Stage1: TEST plots (ROC/Calibration/DCA with CI)
    tags = [f"stage1_{m}" for m in MODEL_LIST]
    log("Stage 1: plotting TEST ROC/Calibration/DCA panels (with CI)...")
    plot_roc_panel_with_ci(tags, "test", os.path.join(FIG_DIR, "stage1_roc_panel_test.svg"))
    plot_calibration_panel_with_ci(tags, "test", os.path.join(FIG_DIR, "stage1_calibration_panel_test.svg"))
    plot_dca_panel_with_ci(tags, "test", os.path.join(FIG_DIR, "stage1_dca_panel_test.svg"))

    log("Stage 1: done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(force=args.force)
