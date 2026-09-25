#!/usr/bin/env python3
"""
figure7_violin_distributions.py — Does MechLib narrow the distribution, or just shift it?
============================================================================================
Violin plots comparing the RESIDUAL distribution (observed - equilibrium)
for AMBER-fixed vs MechLib-corrected equilibrium, for several key protein
backbone observables. A narrower MechLib violin indicates the correction
is capturing real, systematic conformation-dependent shifts (making
predicted geometry more "relaxed"/accurate), not just moving the mean.

Usage:
  python figure7_violin_distributions.py --csv features_lj_FINAL_CLEAN.csv --out ./figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechlib_style import setup_style, COLORS, clean_observable_name

AMBER = {
    'tau_deg':        {'eq': 110.10, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 109.70, 'k': 80.0,  'unit': 'deg'},
    'angle_C_CA_CB':  {'eq': 111.10, 'k': 63.0,  'unit': 'deg'},
    'bond_CA_CB':     {'eq': 1.526, 'k': 317.0, 'unit': 'Å'},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='./figures')
    ap.add_argument('--min_count', type=int, default=10)
    ap.add_argument('--sample', type=int, default=200000,
                    help='Subsample size for violin plot rendering (full dataset is slow to render)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_style()
    import matplotlib.pyplot as plt

    print("Loading CSV...")
    df = pd.read_csv(args.csv, low_memory=False)

    phi_bins = np.arange(-180, 190, 10)
    psi_bins = np.arange(-180, 190, 10)
    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int); sub['qb'] = sub['qb'].astype(int)

    observables = list(AMBER.keys())
    residuals_amber = {}
    residuals_lib = {}

    for col in observables:
        eq = AMBER[col]['eq']
        valid = sub[col].notna()
        sv = sub.loc[valid]
        lib_mean = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('mean')
        counts = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('count')
        pooled_mean = sv.groupby(['pb', 'qb'])[col].transform('mean')
        lib_eq = np.where(counts >= args.min_count, lib_mean, pooled_mean)

        residuals_amber[col] = (sv[col] - eq).values
        residuals_lib[col] = (sv[col] - lib_eq)

        print(f"  {col}: AMBER residual std={np.std(residuals_amber[col]):.4f}, "
              f"MechLib residual std={np.std(residuals_lib[col]):.4f}  "
              f"(narrowing: {100*(1 - np.std(residuals_lib[col])/np.std(residuals_amber[col])):.1f}%)")

    fig, axes = plt.subplots(1, len(observables), figsize=(4*len(observables), 5.5))
    if len(observables) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for i, col in enumerate(observables):
        ax = axes[i]
        ra = residuals_amber[col]
        rl = residuals_lib[col]
        if len(ra) > args.sample:
            idx = rng.choice(len(ra), args.sample, replace=False)
            ra_plot = ra[idx] if isinstance(ra, np.ndarray) else ra.values[idx]
            rl_plot = rl.values[idx] if hasattr(rl, 'values') else rl[idx]
        else:
            ra_plot, rl_plot = ra, rl

        parts = ax.violinplot([ra_plot, rl_plot], showmeans=True, showextrema=False)
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS['amber'] if j == 0 else COLORS['mechlib'])
            pc.set_alpha(0.6)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['AMBER\n(fixed)', 'MechLib'])
        unit = AMBER[col]['unit']
        ax.set_ylabel(f'Residual ({"°" if unit=="deg" else "Å"})')
        std_a = np.std(ra_plot)
        std_l = np.std(rl_plot)
        narrowing = 100 * (1 - std_l / std_a)
        ax.set_title(f'{clean_observable_name(col)}\n\u2212{narrowing:.0f}% std', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(args.out, 'figure7_violin_distributions.png')
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
