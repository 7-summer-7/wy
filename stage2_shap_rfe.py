# stage2_shap_rfe.py
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
# add this line with the other imports
import shap
from common_config import (
    SEED, TAB_DIR, FIG_DIR, SHAP_DIR,
    CONTINUOUS_COLS, BINARY_COLS,
    get_model, load_processed, log, set_seed,
    save_csv, load_json,
    save_shap_cache, load_shap_cache, pick_explain_background,
    _extract_shap_array, shap_summary_svg
)

MODEL_LIST = ["DT", "RF", "XGB", "SVM", "GBDT"]

def main(force: bool = False):
    set_seed(SEED)

    out_best = os.path.join(TAB_DIR, "shap_rfe_best_features.csv")
    if os.path.exists(out_best) and not force:
        log("Stage 2: outputs exist; skip (use --force to redo).")
        return

    # prerequisites
    test_ci_path = os.path.join(TAB_DIR, "stage1_all_features_test_metrics_ci.csv")
    params_path = os.path.join(TAB_DIR, "stage1_all_features_best_params.csv")
    if not os.path.exists(test_ci_path) or not os.path.exists(params_path):
        raise FileNotFoundError("Stage1 outputs missing. Run stage1_tune_full.py first.")

    X_train, X_test, X_ext, y_train, y_test, y_ext, fn = load_processed()
    feature_names = X_train.columns.tolist()

    # pick Stage1 best model by test AUC median
    df_test = pd.read_csv(test_ci_path)
    best_model = df_test.sort_values("AUC_median", ascending=False).iloc[0]["model"]
    log(f"Stage 2: Stage1 best model by TEST AUC median = {best_model}")

    # load its best_params
    df_params = pd.read_csv(params_path)
    best_params = json.loads(df_params[df_params["model"] == best_model].iloc[0]["best_params"])

    # fit model on full features
    model = get_model(best_model, best_params)
    model.fit(X_train.values, y_train)

    # SHAP once (cache)
    shap_tag = f"stage2_bestmodel_{best_model}_train_shap"
    npz_path = os.path.join(SHAP_DIR, f"{shap_tag}.npz")

    if os.path.exists(npz_path) and not force:
        log("Stage 2: SHAP cache exists; loading...")
        shap_vals, base_vals, X_mat, sidx, fns, extra = load_shap_cache(shap_tag)
        X_explain_df = pd.DataFrame(X_mat, columns=fns)
    else:
        log("Stage 2: computing SHAP once...")
        ex_idx, bg_idx = pick_explain_background(len(X_train), explain_n=min(2000, len(X_train)), background_n=200, seed=SEED)
        X_ex = X_train.iloc[ex_idx]
        X_bg = X_train.iloc[bg_idx]

        if best_model in ["DT","RF","XGB","LGBM","GBDT"]:
            explainer = shap.TreeExplainer(model)
            vals_obj = explainer.shap_values(X_ex)
            shap_vals = _extract_shap_array(vals_obj, class_idx=1)
            exp_val = explainer.expected_value
            exp_val = float(np.ravel(exp_val)[1] if isinstance(exp_val, (list, np.ndarray)) and np.ravel(exp_val).size > 1 else np.ravel(exp_val)[0] if isinstance(exp_val, (list, np.ndarray)) else exp_val)
            base_vals = np.full((shap_vals.shape[0],), exp_val)
        else:
            # SVM: KernelExplainer (smaller)
            cap = min(300, len(X_train))
            ex_idx, bg_idx = pick_explain_background(len(X_train), explain_n=min(cap, len(X_train)), background_n=100, seed=SEED)
            X_ex = X_train.iloc[ex_idx]
            X_bg = X_train.iloc[bg_idx]
            f = lambda z: model.predict_proba(z)[:, 1]
            explainer = shap.KernelExplainer(f, X_bg.values)
            shap_vals = explainer.shap_values(X_ex.values, nsamples=200)
            shap_vals = _extract_shap_array(shap_vals, class_idx=1)
            base_vals = np.full((shap_vals.shape[0],), float(explainer.expected_value))

        save_shap_cache(
            shap_tag,
            shap_values=shap_vals,
            base_values=base_vals,
            data_matrix=X_ex.values,
            feature_names=feature_names,
            sample_indices=ex_idx,
            extra={"model": best_model, "best_params": best_params}
        )
        shap_vals, base_vals, X_mat, sidx, fns, extra = load_shap_cache(shap_tag)
        X_explain_df = pd.DataFrame(X_mat, columns=fns)

    # importance (mean abs shap)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    imp_df = pd.DataFrame({"feature": fns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    save_csv(imp_df, os.path.join(TAB_DIR, "shap_importance.csv"))

    # plots
    shap_summary_svg(shap_vals, X_explain_df, os.path.join(FIG_DIR, "shap_importance_bestmodel"), max_display=30)

    # SHAP-RFE (fixed ranking, no re-SHAP)
    ranked_weak_to_strong = imp_df.sort_values("mean_abs_shap", ascending=True)["feature"].tolist()
    current = feature_names.copy()

    results = []
    best_auc = -np.inf
    best_feats = current.copy()

    from sklearn.metrics import roc_auc_score

    log("Stage 2: SHAP-RFE loop start (fixed ranking, refit each step, record TEST AUC point)...")
    while len(current) >= 2:
        m = get_model(best_model, best_params)
        m.fit(X_train[current].values, y_train)
        p_te = m.predict_proba(X_test[current].values)[:, 1]
        try:
            auc_te = float(roc_auc_score(y_test, p_te))
        except Exception:
            auc_te = np.nan

        results.append({"num_features": len(current), "removed_feature": None, "auc_test": auc_te})
        if not np.isnan(auc_te) and auc_te > best_auc:
            best_auc = auc_te
            best_feats = current.copy()

        if len(current) == 2:
            break

        # remove weakest among current
        to_remove = None
        for f in ranked_weak_to_strong:
            if f in current:
                to_remove = f
                break
        results[-1]["removed_feature"] = to_remove
        current = [f for f in current if f != to_remove]

    rfe_df = pd.DataFrame(results)
    save_csv(rfe_df, os.path.join(TAB_DIR, "shap_rfe_results.csv"))
    save_csv(pd.DataFrame({"feature": best_feats}), out_best)

    log(f"Stage 2: best features = {len(best_feats)} | best TEST AUC={best_auc:.4f}")

    # plot AUC vs num_features
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(rfe_df["num_features"], rfe_df["auc_test"], marker="o", linewidth=1.8)
    plt.xlabel("num_features")
    plt.ylabel("test AUC (point)")
    plt.title("SHAP-RFE: AUC vs num_features")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "auc_vs_num_features.svg"), format="svg")
    plt.close()

    log("Stage 2: done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(force=args.force)
