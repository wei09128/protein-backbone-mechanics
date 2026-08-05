#!/usr/bin/env python3
"""
paper3_08_table1_barplot.py

Converts Table 1 (GAM-based coupling decomposition, ΔR²_coupling per
observable) into a horizontal bar chart with 95% CI error bars, per the
reviewer's Table-by-table suggestion. Values are hardcoded from the
published table (not re-derived from raw data) -- this is a visualization
of an already-finalized result, so no CSV input is needed.

If Table 1's numbers change (e.g. after any further robustness check),
update the DATA list below to match -- this script is not wired to
regenerate the underlying statistics.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# observable, dR2_coupling (%), ci_lo, ci_hi, kind ("angle" or "bond")
DATA = [
    ("bond Cα–C",        0.31, 0.25, 0.39, "bond"),
    ("bond N–Cα",         0.28, 0.20, 0.37, "bond"),
    ("∠C–Cα–Cβ",          0.85, 0.74, 0.99, "angle"),
    ("ω (planarity)",     2.30, 2.08, 2.53, "angle"),
    ("∠N–Cα–Cβ",          2.70, 2.40, 2.94, "angle"),
    ("τ (N–Cα–C)",        2.30, 2.15, 2.56, "angle"),
]

COLORS = {"angle": "#2b6cb0", "bond": "#c05621"}

def main(out_dir="."):
    labels = [d[0] for d in DATA]
    vals = np.array([d[1] for d in DATA])
    lo = np.array([d[1] - d[2] for d in DATA])
    hi = np.array([d[3] - d[1] for d in DATA])
    colors = [COLORS[d[4]] for d in DATA]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(len(DATA))
    ax.barh(y, vals, xerr=[lo, hi], color=colors, capsize=3, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Delta R^2_{coupling}$ (%)")
    ax.set_title(r"$\phi\times\psi$ coupling by backbone observable")

    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS["angle"], label="angle"),
               Patch(color=COLORS["bond"], label="bond length")]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "table1_dR2_barplot.png"), dpi=300)
    print(f"Wrote {os.path.join(out_dir, 'table1_dR2_barplot.png')}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    main(args.out)
