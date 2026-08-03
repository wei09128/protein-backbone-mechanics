#!/usr/bin/env python3
"""
figure6_bibii_before_after.py — Diagnostic proof: chain-break fix cleans the data
====================================================================================
Overlapping density plots of the phosphodiester linkage bond (O3'-P next)
before and after the chain-connectivity fix, split by BI/BII state. Makes
the diagnostic story visually self-evident: the pre-fix distribution shows
a heavy contaminated tail (garbage bond lengths from mis-linked strands),
the post-fix distribution is tight and centered near the true AMBER
equilibrium (1.61 A).

Usage:
  python figure6_bibii_before_after.py \\
      --before dna_features_v2_nonredundant.csv \\
      --after dna_features_v3_clean.csv \\
      --out ./figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechlib_style import setup_style, COLORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True, help='dna_features_v2_nonredundant.csv (pre-fix)')
    ap.add_argument('--after', required=True, help='dna_features_v3_clean.csv (post-fix)')
    ap.add_argument('--out', default='./figures')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_style()
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    print("Loading before/after data...")
    before = pd.read_csv(args.before, low_memory=False)
    after = pd.read_csv(args.after, low_memory=False)
    before = before[before['res_name'].isin(['DA', 'DC', 'DG', 'DT'])].copy()
    after = after[after['res_name'].isin(['DA', 'DC', 'DG', 'DT'])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: full range, showing the contaminated tail
    ax = axes[0]
    b_vals = before['bond_O3_Pnext'].dropna().values
    a_vals = after['bond_O3_Pnext'].dropna().values
    print(f"  Before: n={len(b_vals)}, range=[{b_vals.min():.2f}, {b_vals.max():.2f}], "
          f"std={b_vals.std():.3f}")
    print(f"  After:  n={len(a_vals)}, range=[{a_vals.min():.2f}, {a_vals.max():.2f}], "
          f"std={a_vals.std():.3f}")

    bins = np.linspace(1.0, min(b_vals.max(), 10.0), 100)
    ax.hist(b_vals, bins=bins, alpha=0.5, color=COLORS['amber'],
            label=f'Before fix (std={b_vals.std():.2f} \u00c5)', density=True)
    ax.hist(a_vals, bins=bins, alpha=0.6, color=COLORS['mechlib'],
            label=f'After fix (std={a_vals.std():.3f} \u00c5)', density=True)
    ax.axvline(1.610, color='black', linestyle='--', linewidth=1,
               label='AMBER equilibrium (1.610 \u00c5)')
    ax.set_xlabel("O3'(i)\u2013P(i+1) bond length (\u00c5)")
    ax.set_ylabel('Density')
    ax.set_title('A. Full range \u2014 contaminated tail before fix')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # Panel B: zoomed to physical range, by BI/BII state, after fix only
    ax = axes[1]
    for bibii, color in [('BI', COLORS['mechlib']), ('BII', COLORS['charmm'])]:
        vals = after[(after['bi_bii'] == bibii)]['bond_O3_Pnext'].dropna()
        vals = vals[(vals > 1.4) & (vals < 1.9)]
        if len(vals) < 10:
            continue
        kde = gaussian_kde(vals)
        x = np.linspace(1.4, 1.9, 300)
        ax.plot(x, kde(x), color=color, linewidth=2, label=f'{bibii} (n={len(vals):,})')
        ax.fill_between(x, kde(x), alpha=0.25, color=color)
    ax.axvline(1.610, color='black', linestyle='--', linewidth=1, label='AMBER eq.')
    ax.set_xlabel("O3'(i)\u2013P(i+1) bond length (\u00c5)")
    ax.set_ylabel('Density')
    ax.set_title('B. After fix, by BI/BII state (physical range)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(args.out, 'figure6_bibii_before_after.png')
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
