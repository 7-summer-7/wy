import os
import json
import pickle
import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---- import project config/helpers (supports both naming styles) ----
try:
    from common_config import (
        SEED, CLUST_DIR, TAB_DIR, MOD_DIR, DAT_DIR, SITE_COL,
        CONTINUOUS_COLS, BINARY_COLS,
        KPROTO_K_LIST, KPROTO_MAX_ITER, KPROTO_N_INIT,
        load_processed, log, set_seed, save_csv,
        compute_metrics
    )
except Exception:
    # fallback for this codebase structure
    from config import (
        SEED, OUT_DIR, FIG_DIR, DATA_PATH, target_col as TARGET_COL,
        continuous_cols as CONTINUOUS_COLS,
        binary_cols as BINARY_COLS,
        categorical_cols as CATEGORICAL_COLS,
    )
    CLUST_DIR = os.path.join(OUT_DIR, "clusters")
    TAB_DIR = os.path.join(OUT_DIR, "tables")
    MOD_DIR = os.path.join(OUT_DIR, "models")
    DAT_DIR = os.path.join(OUT_DIR, "data_exports")
    SITE_COL = CATEGORICAL_COLS[0] if len(CATEGORICAL_COLS) else "site"
    KPROTO_K_LIST = list(range(2, 6))
    KPROTO_MAX_ITER = 30
    KPROTO_N_INIT = 5

    from common import set_seed, save_csv, log, compute_metrics  # type: ignore

    def load_processed():
        """
        Load processed matrices saved by stage0_preprocess.py
        Expected: X_train_final_std.csv / X_test_final_std.csv / X_ext_final_std.csv
                  y_train.npy / y_test.npy / y_ext.npy
        """
        X_train = pd.read_csv(os.path.join(DAT_DIR, "X_train_final_std.csv"))
        X_test  = pd.read_csv(os.path.join(DAT_DIR, "X_test_final_std.csv"))
        X_ext_p = os.path.join(DAT_DIR, "X_ext_final_std.csv")
        X_ext = pd.read_csv(X_ext_p) if os.path.exists(X_ext_p) else pd.DataFrame(columns=X_train.columns)

        y_train = np.load(os.path.join(DAT_DIR, "y_train.npy"))
        y_test  = np.load(os.path.join(DAT_DIR, "y_test.npy"))
        y_ext_p = os.path.join(DAT_DIR, "y_ext.npy")
        y_ext = np.load(y_ext_p) if os.path.exists(y_ext_p) else np.array([], dtype=int)

        feature_names = X_train.columns.tolist()
        return X_train, X_test, X_ext, y_train, y_test, y_ext, feature_names


# --------------------------
# Controls
# --------------------------
SIL_SAMPLE_MAX = 1500
HEATMAP_SAMPLE_MAX = None  # set to int (e.g., 1500) to cap samples for heatmap
PCA_SCATTER_ALPHA = 1
PCA_SCATTER_SIZE = 22
SEED=3037
#聚4类效果比较好
#3047把IgE聚出来了，其他的一般
#20020110 #其他都很好，只有site4在低风险组
#2026123还可以
# --------------------------
# Feature policy
# cluster features = best subset + dermoscopy features (BINARY_COLS excluding sex/amt)
# --------------------------
EXCLUDE_BINARY_NON_DERM = {"Sex"}  # NOT dermoscopy
DERMOSCOPY_COLS = [c for c in BINARY_COLS if c not in EXCLUDE_BINARY_NON_DERM]


# --------------------------
# distances for K-prototypes (numeric + categorical)
# --------------------------
def _hamming_count(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a != b).astype(float)

def _pairwise_mixed_distance(X_num: np.ndarray, X_cat: np.ndarray, gamma: float) -> np.ndarray:
    """
    D(i,j) = ||x_i - x_j||^2 + gamma * Hamming(cat_i, cat_j)
    """
    X = X_num.astype(np.float64)
    G = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    D_num = sq + sq.T - 2.0 * G
    D_num[D_num < 0] = 0.0

    if X_cat.ndim == 1:
        Xc = X_cat.reshape(-1, 1)
    else:
        Xc = X_cat
    n, m = Xc.shape
    D_cat = np.zeros((n, n), dtype=np.float64)
    for j in range(m):
        col = Xc[:, j].reshape(-1, 1)
        D_cat += (col != col.T).astype(np.float64)

    return D_num + gamma * D_cat


# --------------------------
# K-prototypes core
# --------------------------
def kprototypes_fit_predict(X_num, X_cat, k, gamma, max_iter, n_init, seed):
    rng = np.random.default_rng(seed)
    n = X_num.shape[0]
    best_cost = np.inf
    best_labels = None
    best_cent_num = None
    best_cent_cat = None

    for _ in range(n_init):
        init_idx = rng.choice(n, size=k, replace=False)
        cent_num = X_num[init_idx].copy()
        cent_cat = X_cat[init_idx].copy()

        for _it in range(max_iter):
            d_num = np.sum((X_num[:, None, :] - cent_num[None, :, :]) ** 2, axis=2)
            d_cat = np.sum(_hamming_count(X_cat[:, None, :], cent_cat[None, :, :]), axis=2)
            labels = np.argmin(d_num + gamma * d_cat, axis=1)

            new_num = cent_num.copy()
            new_cat = cent_cat.copy()

            for j in range(k):
                m = labels == j
                if m.sum() == 0:
                    ridx = rng.integers(0, n)
                    new_num[j] = X_num[ridx]
                    new_cat[j] = X_cat[ridx]
                else:
                    new_num[j] = X_num[m].mean(axis=0)
                    for c in range(X_cat.shape[1]):
                        vals, cnt = np.unique(X_cat[m, c], return_counts=True)
                        new_cat[j, c] = vals[np.argmax(cnt)]

            if np.allclose(new_num, cent_num) and np.all(new_cat == cent_cat):
                cent_num, cent_cat = new_num, new_cat
                break

            cent_num, cent_cat = new_num, new_cat

        d_num = np.sum((X_num - cent_num[labels]) ** 2, axis=1)
        d_cat = np.sum(_hamming_count(X_cat, cent_cat[labels]), axis=1)
        cost = float(np.sum(d_num + gamma * d_cat))

        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
            best_cent_num = cent_num.copy()
            best_cent_cat = cent_cat.copy()

    info = {
        "k": int(k),
        "gamma": float(gamma),
        "centroids_num": best_cent_num,
        "centroids_cat": best_cent_cat
    }
    return best_labels, best_cost, info

def kprototypes_predict(X_num, X_cat, info):
    cent_num = info["centroids_num"]
    cent_cat = info["centroids_cat"]
    gamma = float(info["gamma"])
    d_num = np.sum((X_num[:, None, :] - cent_num[None, :, :]) ** 2, axis=2)
    d_cat = np.sum(_hamming_count(X_cat[:, None, :], cent_cat[None, :, :]), axis=2)
    return np.argmin(d_num + gamma * d_cat, axis=1)

def kprototypes_cost(X_num, X_cat, labels, info):
    """
    Compute total cost for a given labeling under info centroids.
    Useful for cohort2 stability: how well cohort2 fits cohort1 centroids.
    """
    cent_num = info["centroids_num"]
    cent_cat = info["centroids_cat"]
    gamma = float(info["gamma"])
    d_num = np.sum((X_num - cent_num[labels]) ** 2, axis=1)
    d_cat = np.sum(_hamming_count(X_cat, cent_cat[labels]), axis=1)
    return float(np.sum(d_num + gamma * d_cat))

def pick_elbow_k(k_list, costs):
    ks = np.array(k_list, dtype=float)
    cs = np.array(costs, dtype=float)
    ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    cs_n = (cs - cs.min()) / (cs.max() - cs.min() + 1e-12)

    p1 = np.array([ks_n[0], cs_n[0]])
    p2 = np.array([ks_n[-1], cs_n[-1]])
    line = p2 - p1
    norm = np.linalg.norm(line) + 1e-12

    dists = []
    for x, y in zip(ks_n, cs_n):
        p = np.array([x, y])
        dist = np.abs(np.cross(line, p - p1)) / norm
        dists.append(dist)

    return k_list[int(np.argmax(dists))]

def pick_composite_k(k_list, costs, sils, w_cost=0.5, w_sil=0.5):
    cs = np.array(costs, dtype=float)
    ss = np.array(sils, dtype=float)
    c_n = (cs - np.nanmin(cs)) / (np.nanmax(cs) - np.nanmin(cs) + 1e-12)
    s_n = (ss - np.nanmin(ss)) / (np.nanmax(ss) - np.nanmin(ss) + 1e-12)
    score = w_cost * (1 - c_n) + w_sil * s_n
    return k_list[int(np.nanargmax(score))]


# --------------------------
# plotting helpers
# --------------------------
def make_cluster_colormap(labels: np.ndarray):
    uniq = np.unique(labels)
    palette = sns.color_palette("Set2", n_colors=len(uniq))
    return {int(c): palette[i] for i, c in enumerate(uniq)}

def plot_k_selection(k_list, costs, sils, elbow_k, sil_k, final_k, out_svg):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_list, costs, marker="o", linewidth=1.8, label="Cost")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Cost")
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(k_list, sils, marker="s", linewidth=1.8, label="Silhouette")
    ax2.set_ylabel("Silhouette")

    ax1.axvline(elbow_k, linestyle="--", linewidth=1.0)
    ax1.text(elbow_k, min(costs), f" elbow={elbow_k}", va="bottom")
    ax1.axvline(sil_k, linestyle="--", linewidth=1.0)
    ax1.text(sil_k, min(costs) + (max(costs)-min(costs))*0.05, f" sil_best={sil_k}", va="bottom")
    ax1.axvline(final_k, linestyle="-", linewidth=1.5)
    ax1.text(final_k, min(costs) + (max(costs)-min(costs))*0.10, f" final={final_k}", va="bottom")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("K selection (Cost + Silhouette)")
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()

def plot_pca_scatter(X, labels, color_map, out_svg, title):
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(7, 5))
    for c in sorted(np.unique(labels).astype(int).tolist()):
        idx = labels == c
        plt.scatter(
            Z[idx, 0], Z[idx, 1],
            s=22,
            alpha=0.75,
            color=color_map[int(c)],
            label=f"Cluster {int(c)}"
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()

def plot_event_rate_bar(labels, y, color_map, out_svg, title="Event rate by cluster"):
    clusters = sorted(np.unique(labels).astype(int).tolist())
    rates, ns = [], []
    for c in clusters:
        idx = labels == c
        ns.append(int(idx.sum()))
        rates.append(float(np.mean(y[idx])) if idx.sum() else np.nan)

    plt.figure(figsize=(7, 5))
    x = np.arange(len(clusters))
    colors = [color_map[int(c)] for c in clusters]
    plt.bar(x, rates, color=colors)
    plt.xticks(x, [str(c) for c in clusters])
    plt.ylabel("Event rate")
    plt.title(title)
    for i, (r, n) in enumerate(zip(rates, ns)):
        if np.isfinite(r):
            plt.text(i, r + 0.01, f"n={n}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()


# --------------------------
# Sample-level heatmap (paper-like)
# --------------------------
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import gridspec

def plot_sample_heatmap_mimic(
    X_std_df: pd.DataFrame,
    labels: np.ndarray,
    continuous_cols: list[str],
    binary_cols: list[str],
    out_svg: str,
    title: str,
    max_n: int | None = None,
    seed: int = 20251221,
    color_map: dict[int, tuple] | None = None,
):
    """
    Paper-like sample heatmap:
      - columns: samples ordered by cluster then within-cluster by PC1
      - rows: features clustered with dendrogram (on continuous block if possible)
      - top: cluster color bar
      - continuous: z-score (RdBu_r)
      - binary: greys (0/1)
    """
    rng = np.random.default_rng(seed)

    cont = [c for c in continuous_cols if c in X_std_df.columns]
    bino = [c for c in binary_cols if c in X_std_df.columns]
    if len(cont) == 0 and len(bino) == 0:
        raise ValueError("No continuous/binary columns found for heatmap.")

    n = X_std_df.shape[0]
    idx_all = np.arange(n)
    if max_n is not None and n > max_n:
        idx_keep = rng.choice(idx_all, size=max_n, replace=False)
        X_plot = X_std_df.iloc[idx_keep].copy()
        lab_plot = labels[idx_keep].copy()
    else:
        X_plot = X_std_df.copy()
        lab_plot = labels.copy()

    # order by cluster then PC1 within cluster
    if len(cont) > 0:
        Z1 = PCA(n_components=1, random_state=seed).fit_transform(X_plot[cont].values).reshape(-1)
        order = np.lexsort((Z1, lab_plot))
    else:
        order = np.argsort(lab_plot)

    X_plot = X_plot.iloc[order]
    lab_plot = lab_plot[order]

    # continuous display z-score across samples (stabilize)
    Xc = None
    if len(cont) > 0:
        Xc = X_plot[cont].astype(float)
        Xc = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + 1e-12)
        Xc = Xc.clip(-4, 4)

    Xb = None
    if len(bino) > 0:
        Xb = X_plot[bino].astype(float).clip(0, 1)

    # row clustering on continuous if exists, else binary
    if Xc is not None:
        row_data = Xc.values.T
        row_names = cont
    else:
        row_data = Xb.values.T
        row_names = bino

    Zlink = linkage(row_data, method="average", metric="euclidean")
    dendro = dendrogram(Zlink, no_plot=True)
    row_order = dendro["leaves"]
    row_names_ord = [row_names[i] for i in row_order]

    if Xc is not None:
        Xc = Xc[row_names_ord]
        cont_ord = row_names_ord
    else:
        bino = row_names_ord
        Xb = Xb[bino]

    # cluster colors
    uniq = np.unique(lab_plot)
    if color_map is None:
        palette = sns.color_palette("Set2", n_colors=len(uniq))
        color_map = {int(c): palette[i] for i, c in enumerate(uniq)}
    top_colors = np.array([color_map[int(c)] for c in lab_plot]).reshape(1, -1, 3)

    n_cont = 0 if Xc is None else Xc.shape[1]
    n_bin = 0 if Xb is None else Xb.shape[1]

    width = max(10, 0.02 * X_plot.shape[0] + 6)
    height = 3 + 0.18 * n_cont + 0.12 * n_bin
    fig = plt.figure(figsize=(width, height))

    gs = gridspec.GridSpec(
        nrows=3 if (Xc is not None and Xb is not None) else 2,
        ncols=2,
        width_ratios=[1.8, 12],
        height_ratios=([0.6, 3.5, 2.0] if (Xc is not None and Xb is not None) else [0.6, 5.5]),
        wspace=0.02,
        hspace=0.10
    )

    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.imshow(top_colors, aspect="auto")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title(title, pad=8)

    ax_den = fig.add_subplot(gs[1, 0])
    dendrogram(Zlink, orientation="left", color_threshold=None, no_labels=True, ax=ax_den)
    ax_den.invert_yaxis()
    ax_den.axis("off")

    ax_h1 = fig.add_subplot(gs[1, 1])
    if Xc is not None:
        im1 = ax_h1.imshow(Xc.values.T, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-4, vmax=4)
        ax_h1.set_yticks(np.arange(n_cont))
        ax_h1.set_yticklabels(cont_ord, fontsize=8)
    else:
        im1 = ax_h1.imshow(Xb.values.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
        ax_h1.set_yticks(np.arange(n_bin))
        ax_h1.set_yticklabels(bino, fontsize=8)

    ax_h1.set_xticks([])
    ax_h1.tick_params(axis="y", length=0)

    # vertical separators between clusters
    last = lab_plot[0]
    for i in range(1, len(lab_plot)):
        if lab_plot[i] != last:
            ax_h1.axvline(i - 0.5, color="black", linewidth=0.6)
            last = lab_plot[i]

    # colorbar for continuous or binary
    cax1 = fig.add_axes([0.92, 0.60, 0.015, 0.22])
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("Z-score" if Xc is not None else "Binary", rotation=90)

    if Xc is not None and Xb is not None:
        ax_h2 = fig.add_subplot(gs[2, 1])
        im2 = ax_h2.imshow(Xb.values.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
        ax_h2.set_yticks(np.arange(n_bin))
        ax_h2.set_yticklabels(Xb.columns.tolist(), fontsize=8)
        ax_h2.set_xticks([])
        ax_h2.tick_params(axis="y", length=0)

        last = lab_plot[0]
        for i in range(1, len(lab_plot)):
            if lab_plot[i] != last:
                ax_h2.axvline(i - 0.5, color="black", linewidth=0.6)
                last = lab_plot[i]

        cax2 = fig.add_axes([0.92, 0.30, 0.015, 0.15])
        cb2 = fig.colorbar(im2, cax=cax2)
        cb2.set_label("Binary (0/1)", rotation=90)

        ax_blank = fig.add_subplot(gs[2, 0])
        ax_blank.axis("off")

    handles = [plt.Line2D([0], [0], color=color_map[int(c)], lw=6) for c in uniq]
    labels_leg = [f"Cluster {int(c)}" for c in uniq]
    ax_top.legend(handles, labels_leg, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


# --------------------------
# profiles table (clinical style)
# --------------------------
def _fmt_count_pct(count: int, total: int) -> str:
    pct = 100.0 * count / total if total else np.nan
    return f"{count} ({pct:.1f}%)" if np.isfinite(pct) else f"{count} (-)"

def _fmt_median_iqr(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "NA"
    q25 = x.quantile(0.25)
    q50 = x.quantile(0.50)
    q75 = x.quantile(0.75)
    return f"{q50:.3g} ({q25:.3g}, {q75:.3g})"

def build_cluster_profiles_table(
    df_cont: pd.DataFrame,
    df_cat: pd.DataFrame,
    labels: np.ndarray,
    y: np.ndarray,
    continuous_cols: list[str],
    categorical_cols: list[str],
    out_csv: str
):
    K = int(np.max(labels)) + 1
    clusters = list(range(K))
    rows = []

    total_all = len(y)
    rows.append({
        "Variable": "Event rate",
        "Overall": f"{float(np.mean(y)):.3f}" if total_all else "NA",
        **{f"Cluster {c}": f"{float(np.mean(y[labels==c])):.3f}" if (labels==c).sum() else "NA" for c in clusters}
    })

    for col in continuous_cols:
        if col not in df_cont.columns:
            continue
        row = {"Variable": col}
        row["Overall"] = _fmt_median_iqr(df_cont[col])
        for c in clusters:
            idx = labels == c
            row[f"Cluster {c}"] = _fmt_median_iqr(df_cont.loc[idx, col])
        rows.append(row)

    for col in categorical_cols:
        if col not in df_cat.columns:
            continue
        levels = pd.Series(df_cat[col].dropna().unique()).tolist()
        try:
            levels = sorted(levels)
        except Exception:
            levels = sorted([str(x) for x in levels])

        for lv in levels:
            row = {"Variable": f"{col}={lv}"}
            all_ser = df_cat[col]
            total = int(all_ser.notna().sum())
            count = int((all_ser == lv).sum())
            row["Overall"] = _fmt_count_pct(count, total)

            for c in clusters:
                idx = labels == c
                ser = df_cat.loc[idx, col]
                total_c = int(ser.notna().sum())
                count_c = int((ser == lv).sum())
                row[f"Cluster {c}"] = _fmt_count_pct(count_c, total_c)
            rows.append(row)

    out = pd.DataFrame(rows)
    save_csv(out, out_csv)
    return out


# --------------------------
# stability report helpers
# --------------------------
def _safe_prob(p: np.ndarray):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum() if p.sum() > 0 else p
    return p

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    JSD(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m), m=(p+q)/2
    bounded, symmetric.
    """
    p = _safe_prob(p)
    q = _safe_prob(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))

def build_cluster_stability_report(
    X_c1_num: np.ndarray,
    X_c1_cat: np.ndarray,
    y_c1: np.ndarray,
    lab_c1: np.ndarray,
    X_c2_num: np.ndarray,
    X_c2_cat: np.ndarray,
    y_c2: np.ndarray,
    lab_c2: np.ndarray,
    info_best: dict,
    gamma: float,
    out_csv: str,
    seed: int = 3047
):
    """
    Outputs a single CSV with:
    - global stability metrics
    - per-cluster size/event rate comparisons
    - centroid drift and categorical mode consistency
    """
    rows = []

    k = int(info_best["k"])
    cent_num = info_best["centroids_num"]
    cent_cat = info_best["centroids_cat"]

    # ---------- global costs ----------
    cost_c1 = kprototypes_cost(X_c1_num, X_c1_cat, lab_c1, info_best)
    cost_c2 = kprototypes_cost(X_c2_num, X_c2_cat, lab_c2, info_best) if X_c2_num.shape[0] else np.nan

    rows.append({
        "scope": "global",
        "metric": "cost_total",
        "cohort1": cost_c1,
        "cohort2": cost_c2
    })
    rows.append({
        "scope": "global",
        "metric": "cost_mean_per_sample",
        "cohort1": cost_c1 / max(1, X_c1_num.shape[0]),
        "cohort2": (cost_c2 / max(1, X_c2_num.shape[0])) if np.isfinite(cost_c2) else np.nan
    })

    # ---------- global silhouette on cohort2 (mixed distance, sample capped) ----------
    if X_c2_num.shape[0] >= 2 and len(np.unique(lab_c2)) >= 2:
        rng = np.random.default_rng(seed)
        idx = np.arange(X_c2_num.shape[0])
        if X_c2_num.shape[0] > SIL_SAMPLE_MAX:
            idx = rng.choice(X_c2_num.shape[0], size=SIL_SAMPLE_MAX, replace=False)
        D2 = _pairwise_mixed_distance(X_c2_num[idx], X_c2_cat[idx], gamma=gamma)
        np.fill_diagonal(D2, 0.0)
        sil2 = float(silhouette_score(D2, lab_c2[idx], metric="precomputed"))
    else:
        sil2 = np.nan

    # cohort1 silhouette（可选，保持对称）
    if X_c1_num.shape[0] >= 2 and len(np.unique(lab_c1)) >= 2:
        rng = np.random.default_rng(seed)
        idx1 = np.arange(X_c1_num.shape[0])
        if X_c1_num.shape[0] > SIL_SAMPLE_MAX:
            idx1 = rng.choice(X_c1_num.shape[0], size=SIL_SAMPLE_MAX, replace=False)
        D1 = _pairwise_mixed_distance(X_c1_num[idx1], X_c1_cat[idx1], gamma=gamma)
        np.fill_diagonal(D1, 0.0)
        sil1 = float(silhouette_score(D1, lab_c1[idx1], metric="precomputed"))
    else:
        sil1 = np.nan

    rows.append({
        "scope": "global",
        "metric": "silhouette_mixed",
        "cohort1": sil1,
        "cohort2": sil2
    })

    # ---------- distribution shift ----------
    c1_counts = pd.Series(lab_c1).value_counts().sort_index()
    c2_counts = pd.Series(lab_c2).value_counts().sort_index() if len(lab_c2) else pd.Series(dtype=int)

    clusters = list(range(k))
    p1 = np.array([c1_counts.get(c, 0) for c in clusters], dtype=float)
    p2 = np.array([c2_counts.get(c, 0) for c in clusters], dtype=float) if len(lab_c2) else np.zeros(k, dtype=float)

    p1n = p1 / p1.sum() if p1.sum() else p1
    p2n = p2 / p2.sum() if p2.sum() else p2

    l1 = float(np.sum(np.abs(p1n - p2n)))
    jsd = float(jensen_shannon_divergence(p1n, p2n)) if (p1n.sum() and p2n.sum()) else np.nan

    rows.append({"scope": "global", "metric": "cluster_prop_L1", "cohort1": 0.0, "cohort2": l1})
    rows.append({"scope": "global", "metric": "cluster_prop_JSD", "cohort1": 0.0, "cohort2": jsd})

    # ---------- per-cluster comparisons ----------
    for c in clusters:
        idx1 = lab_c1 == c
        n1 = int(idx1.sum())
        er1 = float(np.mean(y_c1[idx1])) if n1 else np.nan
        prop1 = float(n1 / max(1, len(lab_c1)))

        if len(lab_c2):
            idx2 = lab_c2 == c
            n2 = int(idx2.sum())
            er2 = float(np.mean(y_c2[idx2])) if n2 else np.nan
            prop2 = float(n2 / max(1, len(lab_c2)))
        else:
            n2, er2, prop2 = 0, np.nan, np.nan

        rows.append({
            "scope": f"cluster_{c}",
            "metric": "n",
            "cohort1": n1,
            "cohort2": n2
        })
        rows.append({
            "scope": f"cluster_{c}",
            "metric": "proportion",
            "cohort1": prop1,
            "cohort2": prop2
        })
        rows.append({
            "scope": f"cluster_{c}",
            "metric": "event_rate",
            "cohort1": er1,
            "cohort2": er2
        })

        # numeric centroid drift: compare cohort2 assigned samples' mean to cohort1 centroid
        if len(lab_c2) and n2 >= 3:
            mu2 = X_c2_num[idx2].mean(axis=0)
            drift_l2 = float(np.linalg.norm(mu2 - cent_num[c]))
        else:
            drift_l2 = np.nan

        rows.append({
            "scope": f"cluster_{c}",
            "metric": "numeric_centroid_drift_L2",
            "cohort1": 0.0,
            "cohort2": drift_l2
        })

        # categorical mode consistency (site): centroid_cat holds cohort1 mode for site
        # here X_cat has shape (n,1) with decoded site category values
        if len(lab_c2) and n2 >= 3:
            vals, cnt = np.unique(X_c2_cat[idx2, 0], return_counts=True)
            mode2 = vals[np.argmax(cnt)]
            consistent = float(mode2 == cent_cat[c, 0])
        else:
            consistent = np.nan

        rows.append({
            "scope": f"cluster_{c}",
            "metric": "site_mode_matches_centroid",
            "cohort1": 1.0,   # cohort1 centroid is defined from cohort1, so treat as matched
            "cohort2": consistent
        })

    df = pd.DataFrame(rows)
    save_csv(df, out_csv)
    return df


def main(force: bool = False, k_final: int | None = None):
    set_seed(SEED)
    os.makedirs(CLUST_DIR, exist_ok=True)

    out_sel = os.path.join(CLUST_DIR, "k_selection_summary.csv")
    out_plot = os.path.join(CLUST_DIR, "k_selection_cost_silhouette.svg")
    out_stability = os.path.join(CLUST_DIR, "cluster_stability_report.csv")

    if os.path.exists(out_sel) and not force:
        log("Stage 6: outputs exist; skip (use --force to redo).")
        return

    bf_path = os.path.join(TAB_DIR, "shap_rfe_best_features.csv")
    if not os.path.exists(bf_path):
        raise FileNotFoundError("Missing best features. Run Stage2 first.")
    best_features = pd.read_csv(bf_path)["feature"].tolist()

    X_train, X_test, X_ext, y_train, y_test, y_ext, fn = load_processed()

    # ============================================================
    # Cohort1 = Train + Test  (按你要求合并成队列1)
    # ============================================================
    X_c1 = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_c1 = np.concatenate([y_train, y_test], axis=0)

    # ---- clustering feature set:
    # best subset (continuous only) + dermoscopy binary (always)
    cluster_features_num = []
    cluster_features_num += [c for c in CONTINUOUS_COLS if c in X_c1.columns and c in best_features]
    cluster_features_num += [c for c in DERMOSCOPY_COLS if c in X_c1.columns]

    # unique keep order
    seen = set()
    cluster_features_num = [c for c in cluster_features_num if not (c in seen or seen.add(c))]
    if len(cluster_features_num) == 0:
        raise ValueError("No clustering numeric features found. Check column names.")

    # ---- site onehot columns (用于热图 & PCA 可视化)
    ohe_cols = [c for c in X_c1.columns if c.startswith(f"{SITE_COL}_")]
    if not ohe_cols:
        raise ValueError("No OHE columns for site in processed data.")

    # ---- decode site from OHE for k-prototypes categorical part
    def ohe_to_site(df_proc: pd.DataFrame) -> np.ndarray:
        arr = df_proc[ohe_cols].values
        idx = np.argmax(arr, axis=1)
        cats = []
        for j in idx:
            name = ohe_cols[j]
            try:
                val = name.split("_")[-1]
                cats.append(int(float(val)))
            except Exception:
                cats.append(int(j))
        return np.array(cats).reshape(-1, 1)

    # ---- build cohort matrices
    Xc1_num = X_c1[cluster_features_num].values
    Xc1_cat = ohe_to_site(X_c1)

    Xc2_num = X_ext[cluster_features_num].values if X_ext.shape[0] else np.empty((0, len(cluster_features_num)))
    Xc2_cat = ohe_to_site(X_ext) if X_ext.shape[0] else np.empty((0, 1))

    # gamma heuristic based on cohort1 numeric block
    gamma = float(np.mean(np.std(Xc1_num, axis=0))) if Xc1_num.shape[1] else 1.0

    # ---- silhouette precompute on cohort1 sample
    n_c1 = Xc1_num.shape[0]
    rng = np.random.default_rng(SEED)
    sil_idx = np.arange(n_c1)
    if n_c1 > SIL_SAMPLE_MAX:
        sil_idx = rng.choice(n_c1, size=SIL_SAMPLE_MAX, replace=False)

    log(f"Stage 6: precomputing mixed distance matrix for silhouette on Cohort1 (n={len(sil_idx)}) ...")
    D = _pairwise_mixed_distance(Xc1_num[sil_idx], Xc1_cat[sil_idx], gamma=gamma)
    np.fill_diagonal(D, 0.0)

    # ---- scan K on cohort1
    k_list = KPROTO_K_LIST
    costs, sils = [], []
    cluster_summaries, infos, labels_map = [], {}, {}

    log("Stage 6: scanning K with cost + silhouette on Cohort1...")
    for k in k_list:
        labels, cost, info = kprototypes_fit_predict(
            Xc1_num, Xc1_cat, k, gamma, KPROTO_MAX_ITER, KPROTO_N_INIT, SEED
        )
        labels_map[k] = labels
        infos[k] = info
        costs.append(cost)

        lab_s = labels[sil_idx]
        sil = float(silhouette_score(D, lab_s, metric="precomputed")) if len(np.unique(lab_s)) >= 2 else np.nan
        sils.append(sil)

        sizes = pd.Series(labels).value_counts().sort_index()
        ev = {int(c): float(np.mean(y_c1[labels == c])) if (labels == c).sum() else np.nan for c in range(k)}

        cluster_summaries.append({
            "K": int(k),
            "cost": float(cost),
            "silhouette": float(sil) if np.isfinite(sil) else np.nan,
            "cluster_sizes": json.dumps({int(k2): int(v) for k2, v in sizes.to_dict().items()}, ensure_ascii=False),
            "cluster_event_rates": json.dumps(ev, ensure_ascii=False),
        })
        log(f"  K={k} cost={cost:.2f} silhouette={sil:.3f}")

    elbow_k = pick_elbow_k(k_list, costs)
    sil_best_k = k_list[int(np.nanargmax(np.array(sils, dtype=float)))]
    composite_k = pick_composite_k(k_list, costs, sils, w_cost=0.5, w_sil=0.5)
    final_k = int(k_final) if k_final is not None else int(composite_k)

    sel_df = pd.DataFrame(cluster_summaries)
    sel_df["elbow_k"] = elbow_k
    sel_df["silhouette_best_k"] = sil_best_k
    sel_df["composite_k"] = composite_k
    sel_df["final_k"] = final_k
    save_csv(sel_df, out_sel)
    plot_k_selection(k_list, costs, sils, elbow_k, sil_best_k, final_k, out_plot)

    with open(os.path.join(CLUST_DIR, "k_selection_decision.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "gamma": gamma,
                "cluster_numeric_features": cluster_features_num,
                "cluster_categorical_features": [SITE_COL],
                "k_candidates": k_list,
                "elbow_k": elbow_k,
                "silhouette_best_k": sil_best_k,
                "composite_k": composite_k,
                "final_k": final_k,
            },
            f, ensure_ascii=False, indent=2
        )

    # ---- final fit (实际上已在扫描时得到 labels_map[final_k] 与 infos[final_k])
    log(f"Stage 6: final K-prototypes on Cohort1 with K={final_k} ...")
    info_best = infos[final_k]
    y_cl_c1 = labels_map[final_k]

    # ---- cohort2 stability validation: predict labels with cohort1 centroids
    y_cl_c2 = kprototypes_predict(Xc2_num, Xc2_cat, info_best) if X_ext.shape[0] else np.array([], dtype=int)

    # ---- outputs (按你要求：只输出队列1聚类结果；队列2用于稳定性验证)
    np.save(os.path.join(CLUST_DIR, "cluster_labels_cohort1.npy"), y_cl_c1)
    np.save(os.path.join(CLUST_DIR, "cluster_labels_cohort2.npy"), y_cl_c2)

    with open(os.path.join(CLUST_DIR, "kproto_model.pkl"), "wb") as f:
        pickle.dump(info_best, f)

    # ---- stable colors (from cohort1)
    color_map = make_cluster_colormap(y_cl_c1)

    # ============================================================
    # PCA scatter（其他步骤不变：仍然输出散点图，只是对象变为 cohort1/cohort2）
    # ============================================================
    vis_features = cluster_features_num + ohe_cols
    plot_pca_scatter(
        X_c1[vis_features].values,
        y_cl_c1,
        color_map,
        os.path.join(CLUST_DIR, "cluster_scatter_pca_cohort1.svg"),
        "Cluster scatter (PCA) — Cohort1 (Train+Test)"
    )
    if X_ext.shape[0]:
        plot_pca_scatter(
            X_ext[vis_features].values,
            y_cl_c2,
            color_map,
            os.path.join(CLUST_DIR, "cluster_scatter_pca_cohort2.svg"),
            "Cluster scatter (PCA) — Cohort2 (External, predicted labels)"
        )

    # ============================================================
    # event rate bars（保持：输出柱图，队列1/队列2）
    # ============================================================
    plot_event_rate_bar(
        y_cl_c1, y_c1, color_map,
        os.path.join(CLUST_DIR, "cluster_event_rate_cohort1.svg"),
        "Event rate by cluster — Cohort1"
    )
    if X_ext.shape[0]:
        plot_event_rate_bar(
            y_cl_c2, y_ext, color_map,
            os.path.join(CLUST_DIR, "cluster_event_rate_cohort2.svg"),
            "Event rate by cluster — Cohort2 (External)"
        )

    # ============================================================
    # heatmaps（关键修改：binary block 加入 site onehot 列）
    # ============================================================
    cont_for_heatmap = [c for c in CONTINUOUS_COLS if c in best_features and c in X_c1.columns]
    # dermoscopy binary + site onehot
    bin_for_heatmap = [c for c in DERMOSCOPY_COLS if c in X_c1.columns] + ohe_cols
    # 去重保持顺序
    seen = set()
    bin_for_heatmap = [c for c in bin_for_heatmap if not (c in seen or seen.add(c))]

    plot_sample_heatmap_mimic(
        X_std_df=X_c1[cont_for_heatmap + bin_for_heatmap],
        labels=y_cl_c1,
        continuous_cols=cont_for_heatmap,
        binary_cols=bin_for_heatmap,
        out_svg=os.path.join(CLUST_DIR, "cluster_heatmap_samples_cohort1.svg"),
        title="Cluster heatmap — sample level (Cohort1, standardized)",
        max_n=HEATMAP_SAMPLE_MAX,
        seed=SEED,
        color_map=color_map
    )

    if X_ext.shape[0]:
        plot_sample_heatmap_mimic(
            X_std_df=X_ext[cont_for_heatmap + bin_for_heatmap],
            labels=y_cl_c2,
            continuous_cols=cont_for_heatmap,
            binary_cols=bin_for_heatmap,
            out_svg=os.path.join(CLUST_DIR, "cluster_heatmap_samples_cohort2.svg"),
            title="Cluster heatmap — sample level (Cohort2, standardized)",
            max_n=HEATMAP_SAMPLE_MAX,
            seed=SEED,
            color_map=color_map
        )

    # ============================================================
    # clinical profiles table（保持：用 orig 尺度数据）
    # ============================================================
    Xtr_orig = pd.read_csv(os.path.join(DAT_DIR, "X_train_imp_orig.csv"))
    Xte_orig = pd.read_csv(os.path.join(DAT_DIR, "X_test_imp_orig.csv"))
    Xc1_orig = pd.concat([Xtr_orig, Xte_orig], axis=0).reset_index(drop=True)

    Xc2_orig = pd.read_csv(os.path.join(DAT_DIR, "X_ext_imp_orig.csv")) if X_ext.shape[0] else pd.DataFrame(columns=Xc1_orig.columns)

    # add decoded site category (not onehot) for profiles
    Xc1_orig[SITE_COL] = ohe_to_site(X_c1).reshape(-1)
    if X_ext.shape[0]:
        Xc2_orig[SITE_COL] = ohe_to_site(X_ext).reshape(-1)

    cont_for_profile = [c for c in CONTINUOUS_COLS if c in Xc1_orig.columns and c in best_features]
    cat_for_profile = [c for c in DERMOSCOPY_COLS if c in Xc1_orig.columns] + [SITE_COL]

    build_cluster_profiles_table(
        df_cont=Xc1_orig, df_cat=Xc1_orig, labels=y_cl_c1, y=y_c1,
        continuous_cols=cont_for_profile, categorical_cols=cat_for_profile,
        out_csv=os.path.join(CLUST_DIR, "cluster_profiles_cohort1_clinical.csv")
    )
    if X_ext.shape[0]:
        build_cluster_profiles_table(
            df_cont=Xc2_orig, df_cat=Xc2_orig, labels=y_cl_c2, y=y_ext,
            continuous_cols=cont_for_profile, categorical_cols=cat_for_profile,
            out_csv=os.path.join(CLUST_DIR, "cluster_profiles_cohort2_clinical.csv")
        )

    # ============================================================
    # cluster distribution cohort1 vs cohort2（保持：现在有统计意义，因为 cohort2 是 predict label）
    # ============================================================
    if X_ext.shape[0]:
        c1_counts = pd.Series(y_cl_c1).value_counts().sort_index()
        c2_counts = pd.Series(y_cl_c2).value_counts().sort_index()
        clusters = sorted(set(c1_counts.index.tolist()) | set(c2_counts.index.tolist()))
        c1 = np.array([c1_counts.get(c, 0) for c in clusters], dtype=float)
        c2 = np.array([c2_counts.get(c, 0) for c in clusters], dtype=float)
        c1 = c1 / c1.sum() if c1.sum() else c1
        c2 = c2 / c2.sum() if c2.sum() else c2

        x = np.arange(len(clusters))
        plt.figure(figsize=(7, 5))
        plt.bar(x - 0.15, c1, width=0.3, label="Cohort1", color=[color_map.get(int(c), (0.6,0.6,0.6)) for c in clusters])
        plt.bar(x + 0.15, c2, width=0.3, label="Cohort2", color=[color_map.get(int(c), (0.3,0.3,0.3)) for c in clusters], alpha=0.6)
        plt.xticks(x, [str(c) for c in clusters])
        plt.ylabel("Proportion")
        plt.title("Cluster distribution: Cohort1 vs Cohort2 (Cohort2 predicted labels)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CLUST_DIR, "cluster_distribution_cohort1_vs_cohort2.svg"), format="svg")
        plt.close()

    # ============================================================
    # stacking stability across clusters（保持：按簇评估 stacking）
    # ============================================================
    stack = joblib.load(os.path.join(MOD_DIR, "stage4_stacking", "stacking.joblib"))

    p_c1 = stack.predict_proba(X_c1[best_features].values)[:, 1]
    p_c2 = stack.predict_proba(X_ext[best_features].values)[:, 1] if X_ext.shape[0] else np.array([], dtype=float)

    def metrics_by_cluster(y, p, labels, out_csv, out_svg):
        rows = []
        k = int(np.max(labels)) + 1 if len(labels) else 0
        for c in range(k):
            idx = labels == c
            if idx.sum() < 10:
                continue
            m = compute_metrics(y[idx], p[idx], thr=0.5)
            rows.append({"cluster": int(c), "n": int(idx.sum()), **m})
        dfm = pd.DataFrame(rows)
        save_csv(dfm, out_csv)

        if not dfm.empty:
            plt.figure(figsize=(7, 5))
            cl = dfm["cluster"].tolist()
            plt.bar([str(c) for c in cl], dfm["AUC"].astype(float), color=[color_map.get(int(c), (0.5,0.5,0.5)) for c in cl])
            plt.xlabel("Cluster")
            plt.ylabel("AUC")
            plt.title("Stacking AUC by cluster")
            plt.tight_layout()
            plt.savefig(out_svg, format="svg")
            plt.close()

    metrics_by_cluster(
        y_c1, p_c1, y_cl_c1,
        os.path.join(CLUST_DIR, "stacking_metrics_by_cluster_cohort1.csv"),
        os.path.join(CLUST_DIR, "stacking_auc_by_cluster_cohort1.svg")
    )
    if X_ext.shape[0]:
        metrics_by_cluster(
            y_ext, p_c2, y_cl_c2,
            os.path.join(CLUST_DIR, "stacking_metrics_by_cluster_cohort2.csv"),
            os.path.join(CLUST_DIR, "stacking_auc_by_cluster_cohort2.svg")
        )

    # ============================================================
    # NEW: cluster stability report（按你要求输出 cluster_stability_report.csv）
    # ============================================================
    if X_ext.shape[0]:
        build_cluster_stability_report(
            X_c1_num=Xc1_num,
            X_c1_cat=Xc1_cat,
            y_c1=y_c1,
            lab_c1=y_cl_c1,
            X_c2_num=Xc2_num,
            X_c2_cat=Xc2_cat,
            y_c2=y_ext,
            lab_c2=y_cl_c2,
            info_best=info_best,
            gamma=gamma,
            out_csv=out_stability,
            seed=SEED
        )
    else:
        # 如果没有外部队列，也输出一个空报告文件，保证 pipeline 不报错
        save_csv(pd.DataFrame([{"note": "No external cohort found; stability report not computed."}]), out_stability)

    log("Stage 6: done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute stage6 artifacts")
    ap.add_argument("--k_final", type=int, default=None, help="force final K (e.g., 3)")
    args = ap.parse_args()
    main(force=args.force, k_final=args.k_final)
