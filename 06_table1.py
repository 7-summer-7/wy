# table1_baseline_compare.py
# Generate Table 1 comparing Train / Test / External cohorts:
# - Continuous: median [P25, P75] + Kruskal–Wallis p
# - Binary: report 0 and 1 as n (%) (two rows) + Chi-square / permutation p
# - Site (1/2/3/4): n (%) per level (four rows) + Chi-square / permutation p
# - Outcome y (binary): n (%) for 0 and 1 (two rows) + Chi-square / permutation p

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

# ---- import your project config ----
from common_config import (
    DAT_DIR, TAB_DIR,
    CONTINUOUS_COLS, BINARY_COLS,
    SITE_COL, TARGET_COL,
)

# -----------------------------
# I/O: edit these filenames if yours differ
# -----------------------------
X_TRAIN_RAW_PATH = os.path.join(DAT_DIR, "X_train_raw.csv")
X_TEST_RAW_PATH  = os.path.join(DAT_DIR, "X_test_raw.csv")
X_EXT_RAW_PATH   = os.path.join(DAT_DIR, "X_ext_raw.csv")

Y_TRAIN_PATH = os.path.join(DAT_DIR, "y_train.npy")
Y_TEST_PATH  = os.path.join(DAT_DIR, "y_test.npy")
Y_EXT_PATH   = os.path.join(DAT_DIR, "y_ext.npy")

OUT_TABLE1_CSV = os.path.join(TAB_DIR, "Table1_baseline_train_test_ext.csv")


# -----------------------------
# Helpers: formatting
# -----------------------------
def fmt_median_iqr(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "NA"
    q25 = x.quantile(0.25)
    q50 = x.quantile(0.50)
    q75 = x.quantile(0.75)
    return f"{q50:.2f} [{q25:.2f}, {q75:.2f}]"

def fmt_count_pct(n: int, total: int) -> str:
    if total <= 0:
        return "0 (NA)"
    pct = 100.0 * n / total
    return f"{n} ({pct:.1f}%)"

def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# -----------------------------
# Stats: omnibus tests across 3 groups
# -----------------------------
def kruskal_p(x1: pd.Series, x2: pd.Series, x3: pd.Series) -> float:
    a = pd.to_numeric(x1, errors="coerce").dropna().values
    b = pd.to_numeric(x2, errors="coerce").dropna().values
    c = pd.to_numeric(x3, errors="coerce").dropna().values
    # Need >=2 groups with data
    groups = [g for g in [a, b, c] if len(g) > 0]
    if len(groups) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*groups).pvalue)
    except Exception:
        return np.nan

def chi2_or_perm_p(
    table: np.ndarray,
    n_perm: int = 10000,
    seed: int = 19880216
) -> float:
    """
    Omnibus test for RxC categorical tables across groups.
    Default: Chi-square; if expected counts <5, use permutation p-value
    based on chi-square statistic under label shuffling.
    """
    table = np.asarray(table, dtype=int)
    if table.ndim != 2 or table.shape[1] != 3:
        # expects 3 groups (Train/Test/External)
        return np.nan

    # If any column total is 0 -> cannot test
    if np.any(table.sum(axis=0) == 0):
        return np.nan

    try:
        chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
        exp_min = np.min(exp)
    except Exception:
        return np.nan

    if exp_min >= 5:
        return float(p)

    # Permutation test (label shuffle) for small expected counts
    rng = np.random.default_rng(seed)

    # Reconstruct individual-level labels from the contingency table
    # levels: rows, groups: columns
    row_levels = []
    grp_labels = []
    for r in range(table.shape[0]):
        for g in range(table.shape[1]):
            cnt = int(table[r, g])
            if cnt > 0:
                row_levels.extend([r] * cnt)
                grp_labels.extend([g] * cnt)
    row_levels = np.array(row_levels, dtype=int)
    grp_labels = np.array(grp_labels, dtype=int)

    # observed chi2
    chi2_obs, _, _, _ = stats.chi2_contingency(table, correction=False)

    # permutation distribution
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(grp_labels)
        # rebuild table_perm
        tperm = np.zeros_like(table)
        for r in range(table.shape[0]):
            for g in range(table.shape[1]):
                tperm[r, g] = int(np.sum((row_levels == r) & (perm == g)))
        chi2_perm, _, _, _ = stats.chi2_contingency(tperm, correction=False)
        if chi2_perm >= chi2_obs - 1e-12:
            ge += 1

    return float((ge + 1) / (n_perm + 1))


# -----------------------------
# Build Table 1
# -----------------------------
def build_table1(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    Xex: pd.DataFrame,
    ytr: np.ndarray,
    yte: np.ndarray,
    yex: np.ndarray,
    continuous_cols: List[str],
    binary_cols: List[str],
    site_col: str,
    outcome_name: str = "Outcome",
    site_levels: Optional[List[int]] = None,
) -> pd.DataFrame:

    site_levels = site_levels or [1, 2, 3, 4]

    n_tr, n_te, n_ex = len(Xtr), len(Xte), len(Xex)

    rows: List[Dict[str, str]] = []

    # --- sample size row (optional but common) ---
    rows.append({
        "Variable": "N",
        "Train": str(n_tr),
        "Test": str(n_te),
        "External": str(n_ex),
        "P": ""
    })

    # --- continuous ---
    for col in continuous_cols:
        if col not in Xtr.columns:
            continue
        p = kruskal_p(Xtr[col], Xte[col], Xex[col] if n_ex else pd.Series([], dtype=float))
        rows.append({
            "Variable": col,
            "Train": fmt_median_iqr(Xtr[col]),
            "Test": fmt_median_iqr(Xte[col]),
            "External": fmt_median_iqr(Xex[col]) if n_ex else "NA",
            "P": fmt_p(p)
        })

    # --- binary (report 0 and 1) ---
    for col in binary_cols:
        if col not in Xtr.columns:
            continue

        def bin_counts(df: pd.DataFrame) -> Tuple[int, int, int]:
            s = pd.to_numeric(df[col], errors="coerce")
            total = int(s.notna().sum())
            n1 = int((s == 1).sum())
            n0 = int((s == 0).sum())
            return n0, n1, total

        tr0, tr1, trtot = bin_counts(Xtr)
        te0, te1, tetot = bin_counts(Xte)
        if n_ex:
            ex0, ex1, extot = bin_counts(Xex)
        else:
            ex0, ex1, extot = 0, 0, 0

        # contingency table for p: rows=level(0/1), cols=groups
        table = np.array([
            [tr0, te0, ex0],
            [tr1, te1, ex1],
        ], dtype=int)
        p = chi2_or_perm_p(table)

        # two display rows
        rows.append({
            "Variable": f"{col}=0",
            "Train": fmt_count_pct(tr0, trtot),
            "Test": fmt_count_pct(te0, tetot),
            "External": fmt_count_pct(ex0, extot) if n_ex else "NA",
            "P": fmt_p(p)
        })
        rows.append({
            "Variable": f"{col}=1",
            "Train": fmt_count_pct(tr1, trtot),
            "Test": fmt_count_pct(te1, tetot),
            "External": fmt_count_pct(ex1, extot) if n_ex else "NA",
            "P": ""  # show p once for the variable block
        })

    # --- site (multi-category 1..4) ---
    if site_col in Xtr.columns:
        def cat_counts(df: pd.DataFrame) -> Tuple[Dict[int, int], int]:
            s = pd.to_numeric(df[site_col], errors="coerce")
            total = int(s.notna().sum())
            cnt = {lv: int((s == lv).sum()) for lv in site_levels}
            return cnt, total

        tr_cnt, trtot = cat_counts(Xtr)
        te_cnt, tetot = cat_counts(Xte)
        if n_ex:
            ex_cnt, extot = cat_counts(Xex)
        else:
            ex_cnt, extot = {lv: 0 for lv in site_levels}, 0

        table = np.array([[tr_cnt[lv], te_cnt[lv], ex_cnt[lv]] for lv in site_levels], dtype=int)
        p = chi2_or_perm_p(table)

        # block header (optional)
        rows.append({
            "Variable": f"{site_col} (overall)",
            "Train": "",
            "Test": "",
            "External": "",
            "P": fmt_p(p)
        })
        for i, lv in enumerate(site_levels):
            rows.append({
                "Variable": f"{site_col}={lv}",
                "Train": fmt_count_pct(tr_cnt[lv], trtot),
                "Test": fmt_count_pct(te_cnt[lv], tetot),
                "External": fmt_count_pct(ex_cnt[lv], extot) if n_ex else "NA",
                "P": ""  # p shown at overall row
            })

    # --- outcome y (binary) ---
    def y_counts(y: np.ndarray) -> Tuple[int, int, int]:
        y = np.asarray(y).astype(int)
        total = int(len(y))
        n1 = int(np.sum(y == 1))
        n0 = int(np.sum(y == 0))
        return n0, n1, total

    ytr0, ytr1, ytrtot = y_counts(ytr)
    yte0, yte1, ytetot = y_counts(yte)
    if len(yex):
        yex0, yex1, yextot = y_counts(yex)
    else:
        yex0, yex1, yextot = 0, 0, 0

    table = np.array([
        [ytr0, yte0, yex0],
        [ytr1, yte1, yex1],
    ], dtype=int)
    p = chi2_or_perm_p(table)

    rows.append({
        "Variable": f"{outcome_name}=0",
        "Train": fmt_count_pct(ytr0, ytrtot),
        "Test": fmt_count_pct(yte0, ytetot),
        "External": fmt_count_pct(yex0, yextot) if len(yex) else " " if yextot == 0 else fmt_count_pct(yex0, yextot),
        "P": fmt_p(p)
    })
    rows.append({
        "Variable": f"{outcome_name}=1",
        "Train": fmt_count_pct(ytr1, ytrtot),
        "Test": fmt_count_pct(yte1, ytetot),
        "External": fmt_count_pct(yex1, yextot) if len(yex) else " " if yextot == 0 else fmt_count_pct(yex1, yextot),
        "P": ""
    })

    return pd.DataFrame(rows)


def main():
    # --- Load ---
    X_train_raw = pd.read_csv(X_TRAIN_RAW_PATH)
    X_test_raw  = pd.read_csv(X_TEST_RAW_PATH)
    X_ext_raw   = pd.read_csv(X_EXT_RAW_PATH) if os.path.exists(X_EXT_RAW_PATH) else pd.DataFrame(columns=X_train_raw.columns)

    y_train = np.load(Y_TRAIN_PATH)
    y_test  = np.load(Y_TEST_PATH)
    y_ext   = np.load(Y_EXT_PATH) if os.path.exists(Y_EXT_PATH) else np.array([], dtype=int)

    # --- Ensure site exists; if your raw data uses another naming, map here ---
    if SITE_COL not in X_train_raw.columns:
        raise ValueError(f"Cannot find site column '{SITE_COL}' in X_train_raw.")

    # --- Build Table 1 ---
    table1 = build_table1(
        Xtr=X_train_raw,
        Xte=X_test_raw,
        Xex=X_ext_raw,
        ytr=y_train,
        yte=y_test,
        yex=y_ext,
        continuous_cols=CONTINUOUS_COLS,
        binary_cols=BINARY_COLS,
        site_col=SITE_COL,
        outcome_name=TARGET_COL,
        site_levels=[1, 2, 3, 4],
    )

    os.makedirs(TAB_DIR, exist_ok=True)
    table1.to_csv(OUT_TABLE1_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {OUT_TABLE1_CSV}")


if __name__ == "__main__":
    main()
