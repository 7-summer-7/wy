from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer

from common_config import (
    SEED, DATA_PATH, TARGET_COL, HOSP_COL, SITE_COL,
    CONTINUOUS_COLS, BINARY_COLS, CATEGORICAL_COLS,
    OUT_DIR, DAT_DIR, MOD_DIR, LOG_DIR,
    ensure_columns, save_json, log, set_seed
)

def main(force: bool = False):
    set_seed(SEED)

    out_train = os.path.join(DAT_DIR, "X_train_final_std.csv")
    if os.path.exists(out_train) and not force:
        log("Stage 0: outputs exist; skip (use --force to redo).")
        return

    log("Stage 0: load data...")
    df = pd.read_csv(DATA_PATH)
    for c in [HOSP_COL, TARGET_COL, SITE_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # ca > 10 -> NA (early)
    if "ca" in df.columns:
        df.loc[df["Ca"] > 10, "ca"] = np.nan

    # Cohorts
    df_h1 = df[df[HOSP_COL] == 1].copy()
    df_h2 = df[df[HOSP_COL] == 2].copy()

    if df_h1.empty:
        raise ValueError("No rows with hosp==1.")

    # Drop hosp
    df_h1 = df_h1.drop(columns=[HOSP_COL])
    df_h2 = df_h2.drop(columns=[HOSP_COL]) if not df_h2.empty else df_h2

    need_cols = CONTINUOUS_COLS + BINARY_COLS + CATEGORICAL_COLS + [TARGET_COL]
    ensure_columns(df_h1, need_cols, "df_h1")
    if not df_h2.empty:
        ensure_columns(df_h2, need_cols, "df_h2")

    # y
    y_h1 = pd.to_numeric(df_h1[TARGET_COL], errors="raise").astype(int).values
    X_h1 = df_h1.drop(columns=[TARGET_COL]).copy()

    # split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_h1, y_h1, test_size=0.3, random_state=SEED, stratify=y_h1
    )
        # 保存原始分割数据
    X_train_raw.to_csv(os.path.join(DAT_DIR, "X_train_raw.csv"), index=False)
    X_test_raw.to_csv(os.path.join(DAT_DIR, "X_test_raw.csv"), index=False)
    X_ext_raw=df_h2.drop(columns=[TARGET_COL])
    X_ext_raw.to_csv(os.path.join(DAT_DIR,"X_ext_raw.csv"),index=False)
    # external
    if not df_h2.empty:
        y_ext = pd.to_numeric(df_h2[TARGET_COL], errors="raise").astype(int).values
        X_ext_raw = df_h2.drop(columns=[TARGET_COL]).copy()
    else:
        y_ext = np.array([], dtype=int)
        X_ext_raw = pd.DataFrame(columns=X_h1.columns)

    # numeric blocks
    num_cols = CONTINUOUS_COLS + BINARY_COLS
    for c in CONTINUOUS_COLS:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="coerce")
        X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="coerce")
        if not X_ext_raw.empty:
            X_ext_raw[c] = pd.to_numeric(X_ext_raw[c], errors="coerce")

    # binary cols: already 0/1, no missing; just enforce numeric
    for c in BINARY_COLS:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="raise")
        X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="raise")
        if not X_ext_raw.empty:
            X_ext_raw[c] = pd.to_numeric(X_ext_raw[c], errors="raise")

    # 1) standardize only continuous numeric columns on train
    scaler = StandardScaler()
    X_train_continuous = X_train_raw[CONTINUOUS_COLS].values
    X_test_continuous = X_test_raw[CONTINUOUS_COLS].values
    X_train_continuous_sc = scaler.fit_transform(X_train_continuous)
    X_test_continuous_sc = scaler.transform(X_test_continuous)

    if not X_ext_raw.empty:
        X_ext_continuous = X_ext_raw[CONTINUOUS_COLS].values
        X_ext_continuous_sc = scaler.transform(X_ext_continuous)
    else:
        X_ext_continuous_sc = np.empty((0, len(CONTINUOUS_COLS)))

    # 2) KNN impute on scaled numeric data (only continuous columns)
    imputer = KNNImputer(n_neighbors=5)
    X_train_continuous_sc_imp = imputer.fit_transform(X_train_continuous_sc)
    X_test_continuous_sc_imp = imputer.transform(X_test_continuous_sc)
    X_ext_continuous_sc_imp = imputer.transform(X_ext_continuous_sc) if X_ext_continuous_sc.shape[0] else np.empty((0, len(CONTINUOUS_COLS)))

    # 3) inverse-transform numeric for interpretation exports
    X_train_continuous_imp_orig = scaler.inverse_transform(X_train_continuous_sc_imp)
    X_test_continuous_imp_orig = scaler.inverse_transform(X_test_continuous_sc_imp)
    X_ext_continuous_imp_orig = scaler.inverse_transform(X_ext_continuous_sc_imp) if X_ext_continuous_sc_imp.shape[0] else np.empty((0, len(CONTINUOUS_COLS)))

    # 4) OHE on site (fit train only)
    cat_cols = CATEGORICAL_COLS
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    X_train_cat = X_train_raw[cat_cols].astype(float).values
    X_test_cat = X_test_raw[cat_cols].astype(float).values
    X_train_ohe = ohe.fit_transform(X_train_cat)
    X_test_ohe = ohe.transform(X_test_cat)

    if not X_ext_raw.empty:
        X_ext_cat = X_ext_raw[cat_cols].astype(float).values
        X_ext_ohe = ohe.transform(X_ext_cat)
    else:
        X_ext_ohe = np.empty((0, X_train_ohe.shape[1]))

    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    final_feature_names = CONTINUOUS_COLS + BINARY_COLS + ohe_names

    # 5) 标准化后地最终训练集、内部验证集、外部验证集
    X_train_final = np.hstack([X_train_continuous_sc_imp, X_train_raw[BINARY_COLS].values, X_train_ohe])
    X_test_final = np.hstack([X_test_continuous_sc_imp, X_test_raw[BINARY_COLS].values, X_test_ohe])
    X_ext_final = np.hstack([X_ext_continuous_sc_imp, X_ext_raw[BINARY_COLS].values, X_ext_ohe]) if X_ext_ohe.shape[0] else np.empty((0, X_train_final.shape[1]))

    # Save standard scaled data
    pd.DataFrame(X_train_final, columns=final_feature_names).to_csv(os.path.join(DAT_DIR, "X_train_final_std.csv"), index=False)
    pd.DataFrame(X_test_final, columns=final_feature_names).to_csv(os.path.join(DAT_DIR, "X_test_final_std.csv"), index=False)
    pd.DataFrame(X_ext_final, columns=final_feature_names).to_csv(os.path.join(DAT_DIR, "X_ext_final_std.csv"), index=False)

    # 插补后原始刻度值
    X_train_imp_orig = np.hstack([X_train_continuous_imp_orig, X_train_raw[BINARY_COLS].values, X_train_ohe])
    X_test_imp_orig = np.hstack([X_test_continuous_imp_orig, X_test_raw[BINARY_COLS].values, X_test_ohe])
    X_ext_imp_orig = np.hstack([X_ext_continuous_imp_orig, X_ext_raw[BINARY_COLS].values, X_ext_ohe]) if X_ext_ohe.shape[0] else np.empty((0, X_train_imp_orig.shape[1]))

    pd.DataFrame(X_train_imp_orig, columns=final_feature_names).to_csv(os.path.join(DAT_DIR, "X_train_imp_orig.csv"), index=False)
    pd.DataFrame(X_test_imp_orig, columns=final_feature_names).to_csv(os.path.join(DAT_DIR, "X_test_imp_orig.csv"), index=False)
    pd.DataFrame(X_ext_imp_orig, columns=final_feature_names).to_csv(os.path.join(DAT_DIR, "X_ext_imp_orig.csv"), index=False)

    # Save other data
    pd.DataFrame(X_train_continuous_imp_orig, columns=CONTINUOUS_COLS).to_csv(os.path.join(DAT_DIR, "X_train_continuous_imp_orig.csv"), index=False)
    pd.DataFrame(X_test_continuous_imp_orig, columns=CONTINUOUS_COLS).to_csv(os.path.join(DAT_DIR, "X_test_continuous_imp_orig.csv"), index=False)
    pd.DataFrame(X_ext_continuous_imp_orig, columns=CONTINUOUS_COLS).to_csv(os.path.join(DAT_DIR, "X_ext_continuous_imp_orig.csv"), index=False)
    #
    np.save(os.path.join(DAT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DAT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(DAT_DIR, "y_ext.npy"), y_ext)

    save_json(
        {"final_feature_names": final_feature_names, "ohe_names": ohe_names, "continuous_cols": CONTINUOUS_COLS, "cat_cols": cat_cols},
        os.path.join(DAT_DIR, "feature_names.json")
    )
    save_json(ohe_names, os.path.join(DAT_DIR, "ohe_feature_names.json"))

    # Save preprocessors
    joblib.dump(scaler, os.path.join(MOD_DIR, "preprocess_scaler.joblib"))
    joblib.dump(imputer, os.path.join(MOD_DIR, "preprocess_knn_imputer.joblib"))
    joblib.dump(ohe, os.path.join(MOD_DIR, "preprocess_ohe.joblib"))

    # Save run config (including confirmed global SHAP sampling)
    save_json(
        {
            "SEED": SEED,
            "DATA_PATH": DATA_PATH,
            "TARGET_COL": TARGET_COL,
            "GLOBAL_SHAP_EXPLAIN_N": 500,
            "GLOBAL_SHAP_BACKGROUND_N": 100
        },
        os.path.join(LOG_DIR, "run_config.json")
    )

    log("Stage 0: done. Exports + preprocessors saved.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(force=args.force)
