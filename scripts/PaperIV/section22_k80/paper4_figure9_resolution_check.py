#!/usr/bin/env python3
"""
figure9_resolution_check.py — Is strain reduction independent of resolution?
================================================================================
Per-structure strain reduction vs. PDB resolution. If MechLib's improvement
is uniform across resolution, that argues the effect is a real geometric
regularity, not an artifact of fitting low-resolution coordinate noise.

Usage:
  python figure9_resolution_check.py --csv features_lj_FINAL_CLEAN.csv \\
      --resolutions resolutions.csv --out ./figures/
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

AMBER = {
    'tau_deg':        {'eq': 110.10, 'k': 63.0,  'unit': 'deg'},
    'angle_N_CA_CB':  {'eq': 109.70, 'k': 80.0,  'unit': 'deg'},
    'angle_C_CA_CB':  {'eq': 111.10, 'k': 63.0,  'unit': 'deg'},
    'angle_CaCN':     {'eq': 116.6, 'k': 70.0,  'unit': 'deg'},
    'angle_CNCa':     {'eq': 121.9, 'k': 50.0,  'unit': 'deg'},
    'angle_CA_C_O':   {'eq': 120.4, 'k': 80.0,  'unit': 'deg'},
    'bond_N_CA':      {'eq': 1.449,  'k': 337.0, 'unit': 'Å'},
    'bond_CA_C':      {'eq': 1.522, 'k': 317.0, 'unit': 'Å'},
    'bond_C_O':       {'eq': 1.229, 'k': 570.0, 'unit': 'Å'},
    'bond_C_N_next':  {'eq': 1.335, 'k': 490.0, 'unit': 'Å'},
    'bond_CA_CB':     {'eq': 1.526, 'k': 317.0, 'unit': 'Å'},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--resolutions', required=True)
    ap.add_argument('--out', default='./figures')
    ap.add_argument('--min_count', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_style()
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, binned_statistic

    print("Loading data...")
    df = pd.read_csv(args.csv, low_memory=False)
    res_df = pd.read_csv(args.resolutions)
    res_df = res_df.dropna(subset=['resolution'])
    res_map = dict(zip(res_df['pdb_id'], res_df['resolution']))

    phi_bins = np.arange(-180, 190, 10)
    psi_bins = np.arange(-180, 190, 10)
    sub = df.copy()
    sub['pb'] = pd.cut(sub['phi_deg'], phi_bins, labels=False, right=False)
    sub['qb'] = pd.cut(sub['psi_deg'], psi_bins, labels=False, right=False)
    sub = sub.dropna(subset=['pb', 'qb'])
    sub['pb'] = sub['pb'].astype(int); sub['qb'] = sub['qb'].astype(int)

    geo_cols = [c for c in AMBER if c in sub.columns]
    total_a = np.zeros(len(sub))
    total_l = np.zeros(len(sub))
    for col in geo_cols:
        k = AMBER[col]['k']
        conv = np.pi / 180.0 if AMBER[col]['unit'] == 'deg' else 1.0
        eq = AMBER[col]['eq']
        valid = sub[col].notna()
        sv = sub.loc[valid]
        lib_mean = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('mean')
        counts = sv.groupby(['res_name', 'pb', 'qb'])[col].transform('count')
        pooled_mean = sv.groupby(['pb', 'qb'])[col].transform('mean')
        lib_eq = np.where(counts >= args.min_count, lib_mean, pooled_mean)
        ea = np.zeros(len(sub)); el = np.zeros(len(sub))
        ea[valid.values] = 0.5 * k * ((sv[col] - eq) * conv) ** 2
        el[valid.values] = 0.5 * k * ((sv[col] - lib_eq) * conv) ** 2
        total_a += ea; total_l += el
    sub['_total_a'] = total_a
    sub['_total_l'] = total_l

    print("Aggregating per structure...")
    per_struct = sub.groupby('pdb_id').agg(
        mean_a=('_total_a', 'mean'), mean_l=('_total_l', 'mean')).reset_index()
    per_struct['resolution'] = per_struct['pdb_id'].map(res_map)
    per_struct = per_struct.dropna(subset=['resolution'])
    per_struct['pct_reduction'] = 100 * (per_struct['mean_a'] - per_struct['mean_l']) / per_struct['mean_a']
    per_struct = per_struct[(per_struct['resolution'] > 0) & (per_struct['resolution'] < 4.0)]

    r, p = pearsonr(per_struct['resolution'], per_struct['pct_reduction'])
    print(f"Correlation between resolution and %reduction: r={r:.3f}, p={p:.2e}, n={len(per_struct)}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(per_struct['resolution'], per_struct['pct_reduction'],
               s=4, alpha=0.15, color=COLORS['mechlib'])

    bin_means, bin_edges, _ = binned_statistic(
        per_struct['resolution'], per_struct['pct_reduction'], statistic='median', bins=15)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, bin_means, color=COLORS['amber'], linewidth=2.5,
            marker='o', markersize=5, label='Median per resolution bin')

    ax.set_xlabel('Resolution (Å)')
    ax.set_ylabel('Per-structure strain reduction (%)')
    ax.set_title(f'Strain reduction vs. resolution\n(r = {r:.3f}, p = {p:.2e}, n = {len(per_struct)} structures)')
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(args.out, 'figure9_resolution_check.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
