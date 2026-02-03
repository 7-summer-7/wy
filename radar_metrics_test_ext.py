# radar_metrics_test_ext.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_config import TAB_DIR, FIG_DIR

METRICS = ["AUC", "F1", "ACC", "PRE", "SEN", "SPE"]
ORDER = ["DT", "RF", "XGB", "SVM", "GBDT", "STACK"]
# ----- Radar scale controls -----
R_MIN = 0.00          # 最小刻度（例如 0.5）
R_MAX = 0.85          # 最大刻度（例如 1.0）
R_STEP = 0.10         # 刻度间隔（例如 0.1）

# Okabe–Ito (colorblind-friendly) palette + one extra gray
OKABE_ITO = {
    "DT":   "#0072B2",  # blue
    "RF":   "#E69F00",  # orange
    "XGB":  "#009E73",  # green
    "SVM":  "#CC79A7",  # purple/pink
    "GBDT": "#D55E00",  # vermillion
    "STACK":"#4D4D4D",  # dark gray
}

def _load_and_merge(stage3_csv: str, stage4_csv: str) -> pd.DataFrame:
    df3 = pd.read_csv(stage3_csv)
    df4 = pd.read_csv(stage4_csv)

    # Normalize model name column
    if "model" not in df3.columns:
        raise ValueError(f"'model' column not found in {stage3_csv}")
    if "model" not in df4.columns:
        raise ValueError(f"'model' column not found in {stage4_csv}")

    # Keep only medians
    keep_cols = ["model"] + [f"{m}_median" for m in METRICS]
    df3 = df3[keep_cols].copy()
    df4 = df4[keep_cols].copy()

    # Force stacking row model name
    # (your stage4 file usually already uses "STACK", but keep robust)
    df4["model"] = df4["model"].replace({"Stack": "STACK", "stack": "STACK", "STACKING": "STACK"})

    df = pd.concat([df3, df4], axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["model"], keep="last").set_index("model")

    # Reindex to desired order and drop missing models
    df = df.reindex([m for m in ORDER if m in df.index])

    # Ensure values are numeric and clipped to [0,1]
    for m in METRICS:
        col = f"{m}_median"
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 1)

    return df

def radar_plot(df: pd.DataFrame, title: str, out_svg: str, out_png: str) -> None:
    labels = METRICS
    n = len(labels)

    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.4, 6.6), dpi=200)
    ax = plt.subplot(111, polar=True)

    # start at top, clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # --- x labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    # --- radial scale (zoom in)
    R_MIN, R_MAX, R_STEP = 0.50, 0.90, 0.10
    ax.set_ylim(R_MIN, R_MAX)
    ticks = np.arange(R_MIN, R_MAX + 1e-9, R_STEP)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=9)

    # --- grid styling (subtle)
    ax.grid(True, alpha=0.12, linewidth=0.8)
    ax.spines["polar"].set_alpha(0.25)

    # --- plot: lines for all, fill ONLY for STACK
    for model in df.index:
        vals = [df.loc[model, f"{m}_median"] for m in METRICS]
        vals = [np.nan if pd.isna(v) else float(v) for v in vals]
        vals = np.clip(vals, R_MIN, R_MAX)  # optional: keep within view
        vals = vals.tolist() + [vals[0]]

        color = OKABE_ITO.get(model, "#333333")

        is_stack = (model == "STACK")
        lw = 2.6 if is_stack else 1.8
        z = 4 if is_stack else 2

        ax.plot(
            angles, vals,
            color=color,
            linewidth=lw,
            marker="o",
            markersize=3.8 if is_stack else 3.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=model,
            zorder=z
        )

        if is_stack:
            ax.fill(angles, vals, color=color, alpha=0.10, zorder=1)

    ax.set_title(title, fontsize=13, pad=18)

    # legend: clean and outside
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.03),
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)

def main():
    # ---- TEST radar ----
    test_stage3 = os.path.join(TAB_DIR, "stage3_best_features_test_metrics_ci.csv")
    test_stage4 = os.path.join(TAB_DIR, "stage4_stacking_test_metrics_ci.csv")
    df_test = _load_and_merge(test_stage3, test_stage4)

    radar_plot(
        df_test,
        title="Performance comparison (median) — TEST",
        out_svg=os.path.join(FIG_DIR, "radar_metrics_test.svg"),
        out_png=os.path.join(FIG_DIR, "radar_metrics_test.png"),
    )

    # ---- EXT radar ----
    ext_stage3 = os.path.join(TAB_DIR, "stage3_best_features_ext_metrics_ci.csv")
    ext_stage4 = os.path.join(TAB_DIR, "stage4_stacking_ext_metrics_ci.csv")
    df_ext = _load_and_merge(ext_stage3, ext_stage4)

    radar_plot(
        df_ext,
        title="Performance comparison (median) — EXTERNAL",
        out_svg=os.path.join(FIG_DIR, "radar_metrics_ext.svg"),
        out_png=os.path.join(FIG_DIR, "radar_metrics_ext.png"),
    )

    print("[OK] Saved radar charts to:", FIG_DIR)

if __name__ == "__main__":
    main()
