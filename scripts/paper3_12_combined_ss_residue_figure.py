#!/usr/bin/env python3
"""
paper3_12_combined_ss_residue_figure.py

Combines the Table 2 (secondary structure) and Table 3 (residue class)
conversions into a single two-panel figure, per the feedback doc's
suggested figure order: "Figure 4: Secondary structure + residue-
dependence (current Tables 2+3)". Supersedes running
paper3_09_table2_barplot.py and paper3_10_table3_barplot.py separately
if you want the merged version for the main text; those two scripts
still work standalone if you'd rather keep them as separate figures.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

SS_DATA = [
    ("Coil", 2.34, 2.18, 2.52),
    ("αL",   1.31, 1.23, 1.38),
    ("αR",   1.29, 1.18, 1.37),
    ("β",    0.62, 0.55, 0.76),
    ("PPII", 0.62, 0.53, 0.73),
]

RESIDUE_DATA = [
    ("Non-branched other", 2.42, 2.20, 2.63, "Full φψ access"),
    ("Glycine",            2.09, 1.98, 2.28, "No sidechain"),
    ("β-branched",         1.92, 1.71, 2.11, "Constrained χ¹"),
    ("Proline",            1.16, 1.04, 1.30, "φ locked"),
]

def main(out_dir="."):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 3.4))

    # Panel A: secondary structure
    labels = [d[0] for d in SS_DATA]
    vals = np.array([d[1] for d in SS_DATA])
    lo = np.array([d[1] - d[2] for d in SS_DATA])
    hi = np.array([d[3] - d[1] for d in SS_DATA])
    x = np.arange(len(SS_DATA))
    axA.bar(x, vals, yerr=[lo, hi], capsize=3, color="#2b6cb0", width=0.55)
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_ylabel(r"$\tau$ coupling, $\Delta R^2$ (%)")
    axA.set_title("A. By secondary structure")

    # Panel B: residue class
    labels_b = [d[0] for d in RESIDUE_DATA]
    vals_b = np.array([d[1] for d in RESIDUE_DATA])
    lo_b = np.array([d[1] - d[2] for d in RESIDUE_DATA])
    hi_b = np.array([d[3] - d[1] for d in RESIDUE_DATA])
    mechanisms = [d[4] for d in RESIDUE_DATA]
    y = np.arange(len(RESIDUE_DATA))
    axB.barh(y, vals_b, xerr=[lo_b, hi_b], capsize=3, color="#2b6cb0", height=0.55)
    axB.set_yticks(y)
    axB.set_yticklabels(labels_b)
    axB.set_xlabel(r"$\tau$ coupling, $\Delta R^2$ (%)")
    axB.set_title("B. By residue class")
    axB.set_xlim(0, max(vals_b) + 1.4)
    for yi, ci_hi, mech in zip(y, [d[3] for d in RESIDUE_DATA], mechanisms):
        axB.text(ci_hi + 0.15, yi, mech, va="center", fontsize=7.5, color="#555555")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "figure_ss_residue_combined.png"), dpi=300)
    print(f"Wrote {os.path.join(out_dir, 'figure_ss_residue_combined.png')}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    main(args.out)
