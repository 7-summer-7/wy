# stage_corr_network.py
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from common_config import (
    SEED, DAT_DIR, TAB_DIR, FIG_DIR, log, set_seed, save_csv
)

# -----------------------------
# Config
# -----------------------------
SPLIT = "train"  # "train" / "test" / "ext"
METHOD = "spearman"  # "spearman" or "pearson"
ABS_R_THR = 0.30     # edge threshold on |corr|
TOP_N_EDGES = None   # e.g., 800; if None, keep all above threshold
EXCLUDE_PATTERNS = []  # e.g., [r"^site_"] to exclude site one-hot if you want
MAX_NODES_TO_DRAW_LABELS = 80  # if too many nodes, avoid clutter
LAYOUT = "spring"  # "spring" or "kamada_kawai"
OUT_PREFIX = os.path.join(FIG_DIR, f"corr_network_{SPLIT}_{METHOD}_thr{ABS_R_THR:.2f}")

# -----------------------------
# Utilities
# -----------------------------
def load_split_matrix(split: str) -> pd.DataFrame:
    path = os.path.join(DAT_DIR, f"X_{split}_final_std.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    return df

def drop_by_patterns(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    if not patterns:
        return df
    keep = []
    for c in df.columns:
        if any(re.search(pat, c) for pat in patterns):
            continue
        keep.append(c)
    return df[keep].copy()

def compute_corr(df: pd.DataFrame, method: str) -> pd.DataFrame:
    # ensure numeric
    X = df.apply(pd.to_numeric, errors="coerce")
    # drop constant columns to avoid NaN correlations
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]
    corr = X.corr(method=method)
    return corr

def corr_to_edge_list(corr: pd.DataFrame, abs_thr: float) -> pd.DataFrame:
    cols = corr.columns
    mat = corr.values
    edges = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            r = mat[i, j]
            if np.isnan(r):
                continue
            if abs(r) >= abs_thr:
                edges.append((cols[i], cols[j], float(r), "pos" if r > 0 else "neg", abs(float(r))))
    edf = pd.DataFrame(edges, columns=["source", "target", "corr", "sign", "abs_corr"])
    edf = edf.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return edf

def build_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(r["source"], r["target"], weight=float(abs(r["corr"])), corr=float(r["corr"]), sign=r["sign"])
    return G

def compute_node_stats(G: nx.Graph) -> pd.DataFrame:
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    out = pd.DataFrame({
        "node": list(deg.keys()),
        "degree": [deg[k] for k in deg.keys()],
        "weighted_degree": [wdeg[k] for k in wdeg.keys()],
    })
    out = out.sort_values(["weighted_degree", "degree"], ascending=False).reset_index(drop=True)
    return out

def draw_network(G: nx.Graph, out_svg: str, out_png: str, title: str):
    if G.number_of_nodes() == 0:
        raise ValueError("No edges passed the threshold. Lower ABS_R_THR or increase TOP_N_EDGES.")

    # layout
    if LAYOUT == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=SEED, k=None)

    # node sizes by weighted degree
    wdeg = dict(G.degree(weight="weight"))
    wvals = np.array([wdeg[n] for n in G.nodes()], dtype=float)
    if wvals.size == 0:
        wvals = np.ones((len(G.nodes()),), dtype=float)
    node_sizes = 200 + 900 * (wvals - wvals.min()) / (wvals.max() - wvals.min() + 1e-12)

    # edges: split pos/neg
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign") == "pos"]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("sign") == "neg"]

    # edge widths by abs corr
    widths = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    ew = 0.5 + 4.0 * (widths - widths.min()) / (widths.max() - widths.min() + 1e-12)

    # map width per edge for each subset
    edge_width_map = {(u, v): float(w) for (u, v), w in zip(G.edges(), ew)}
    pos_w = [edge_width_map[(u, v)] if (u, v) in edge_width_map else edge_width_map[(v, u)] for (u, v) in pos_edges]
    neg_w = [edge_width_map[(u, v)] if (u, v) in edge_width_map else edge_width_map[(v, u)] for (u, v) in neg_edges]

    plt.figure(figsize=(12, 10))
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.85)

    # edges (do not set explicit colors; use default for pos, dashed for neg to differentiate)
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=pos_w, alpha=0.35)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=neg_w, alpha=0.35, style="dashed")

    # labels only if not too many nodes
    if G.number_of_nodes() <= MAX_NODES_TO_DRAW_LABELS:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.savefig(out_png, format="png", dpi=300)
    plt.close()

def main(force: bool = False):
    set_seed(SEED)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    out_edges = os.path.join(TAB_DIR, f"corr_network_edges_{SPLIT}_{METHOD}_thr{ABS_R_THR:.2f}.csv")
    out_nodes = os.path.join(TAB_DIR, f"corr_network_nodes_{SPLIT}_{METHOD}_thr{ABS_R_THR:.2f}.csv")
    out_svg = OUT_PREFIX + ".svg"
    out_png = OUT_PREFIX + ".png"

    if (os.path.exists(out_edges) and os.path.exists(out_svg) and (not force)):
        log("Corr network: outputs exist; skip (use force=True to redo).")
        return

    log(f"Corr network: loading split={SPLIT} ...")
    X = load_split_matrix(SPLIT)
    X = drop_by_patterns(X, EXCLUDE_PATTERNS)

    log(f"Corr network: computing correlation ({METHOD}) ...")
    corr = compute_corr(X, method=METHOD)

    log(f"Corr network: building edges with |r| >= {ABS_R_THR} ...")
    edges = corr_to_edge_list(corr, abs_thr=ABS_R_THR)
    if TOP_N_EDGES is not None and len(edges) > TOP_N_EDGES:
        edges = edges.iloc[:TOP_N_EDGES].copy()

    save_csv(edges, out_edges)

    log("Corr network: building graph + node stats ...")
    G = build_graph(edges)
    nodes = compute_node_stats(G)
    save_csv(nodes, out_nodes)

    title = f"Correlation network ({METHOD}), split={SPLIT}, |r|≥{ABS_R_THR}, edges={G.number_of_edges()}, nodes={G.number_of_nodes()}"
    log("Corr network: drawing ...")
    draw_network(G, out_svg=out_svg, out_png=out_png, title=title)

    log(f"Corr network: done. Saved:\n  {out_svg}\n  {out_png}\n  {out_edges}\n  {out_nodes}")

if __name__ == "__main__":
    main(force=False)
