#!/usr/bin/env python3
"""
paper3_11_table5_dotplot.py
Converts Table 5 (QM descriptors do not improve prediction beyond phi/psi)
into a dot plot. Values hardcoded from the published table; see note in
paper3_08 script. Inequality values (<, <=) are plotted at the stated
bound and annotated accordingly -- these are upper bounds, not point
estimates, so no error bars are drawn for them.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# descriptor, dR2_tau, dR2_angles, inequality ("" | "<" | "<=")
DATA = [
    ("n→π* (O···C distance)", 0.006, 0.006, ""),
    ("σ→σ* hyperconjugation", 0.004, 0.004, "<"),
    ("Both combined",         0.02,  0.02,  "≤"),
]

def main(out_dir="."):
    labels = [d[0] for d in DATA]
    tau_vals = [d[1] for d in DATA]
    ang_vals = [d[2] for d in DATA]
    ineq = [d[3] for d in DATA]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    y = np.arange(len(DATA))
    ax.scatter(tau_vals, y + 0.12, marker="o", s=60, color="#2b6cb0", label=r"$\Delta R^2$ for $\tau$")
    ax.scatter(ang_vals, y - 0.12, marker="s", s=60, color="#c05621", label=r"$\Delta R^2$ for angles")

    for yi, tv, sym in zip(y, tau_vals, ineq):
        label = f"{sym}{tv:.3f}" if sym else f"{tv:.3f}"
        ax.text(tv + 0.001, yi + 0.12, label, va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Incremental $\Delta R^2$ beyond $(\phi,\psi)$ binning")
    ax.set_title("QM descriptors are redundant with backbone geometry")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xlim(0, 0.03)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "table5_qm_dotplot.png"), dpi=300)
    print(f"Wrote {os.path.join(out_dir, 'table5_qm_dotplot.png')}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    main(args.out)
